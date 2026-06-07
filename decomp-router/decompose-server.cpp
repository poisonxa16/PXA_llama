// opus-microme DECOMPOSER SERVER — the model embedded in the llama.cpp/ggml engine, in one binary.
// Loads microme-body.gguf (BERT cross-encoder body) + microme-head.bin (2-layer head), serves POST
// /decompose {"prompt":...} -> dependency DAG + fan-out decision. Pure ggml inference, no ONNX, no python.
// Isolated build (separate binary), parity-gated vs the validated sidecar before it ships.
#include "common.h"
#include "arg.h"
#include "log.h"
#include "llama.h"
#include "httplib.h"
#include <string>
#include <vector>
#include <regex>
#include <fstream>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <set>

// ---------------- head (pre_classifier 768x768 + ReLU + classifier 2x768) ----------------
static std::vector<float> PRE_W, PRE_B, CLS_W, CLS_B;
static int NE = 768;
static bool load_head(const std::string & path) {
    std::ifstream f(path, std::ios::binary); if (!f) return false;
    auto rd = [&](std::vector<float> & v, size_t n){ v.resize(n); f.read((char*)v.data(), n*sizeof(float)); };
    rd(PRE_W, (size_t)NE*NE); rd(PRE_B, NE); rd(CLS_W, (size_t)2*NE); rd(CLS_B, 2);
    return (bool)f;
}
static int head_predict(const float * cls) {           // 1 = "later depends on earlier"
    std::vector<float> x(NE);
    for (int i = 0; i < NE; i++) { float s = PRE_B[i]; const float * w = &PRE_W[(size_t)i*NE];
        for (int j = 0; j < NE; j++) s += w[j]*cls[j]; x[i] = s > 0 ? s : 0.0f; }   // ReLU
    float l0 = CLS_B[0], l1 = CLS_B[1];
    for (int j = 0; j < NE; j++) { l0 += CLS_W[j]*x[j]; l1 += CLS_W[NE+j]*x[j]; }
    return l1 > l0 ? 1 : 0;
}

// ---------------- segmenter (faithful port of seg.py) ----------------
static const std::regex TASK_VERB(
 "\\b(write|create|build|make|implement|code|draft|compose|generate|produce|design|refactor|fix|debug|compute|calculate|count|sum|add|multiply|solve|explain|describe|summarize|summarise|outline|define|list|enumerate|name|give|provide|show|translate|convert|fetch|get|read|find|search|look up|pick|choose|select|compare|contrast|analyze|analyse|rewrite|tell|remind|send|email|set|update|say|roll|shuffle|deal|open|total|sort|book|recommend|round|evaluate|double"
 "|escribe|escribir|crea|crear|haz|hacer|implementa|calcula|calcular|explica|explicar|describir|resume|resumir|definir|lista|listar|enumera|nombra|nombrar|dame|dar|muestra|mostrar|traduce|traducir|convierte|convertir|busca|buscar|encuentra|encontrar|lee|leer|elige|elegir|escoge|selecciona|compara|comparar|reescribe|dime|decir|recuerda|envia|enviar|cambia|cambiar|actualiza|suma|sumar|multiplica|ordena|ordenar|reserva|recomienda|redondea|evalua|duplica|genera|generar|pon|poner)\\b",
 std::regex::icase);
static const std::regex NUMLIST("(^|\\s)\\d\\s*[\\.\\)]\\s+");
static const std::string COORD_BASE =
 ";|,?\\s+and\\s+also\\s+|,?\\s+and\\s+separately\\s+|,?\\s+and\\s+independently\\s+|,?\\s+and\\s+then\\s+|,?\\s+then\\s+|,?\\s+after\\s+(?:that|which)\\s+|,?\\s+next\\s+|,?\\s+finally\\s+|,?\\s+also\\s+|,?\\s+plus\\s+"
 "|,?\\s+y\\s+luego\\s+|,?\\s+y\\s+despu[e\xc3\xa9]s\\s+|,?\\s+luego\\s+|,?\\s+despu[e\xc3\xa9]s\\s+|,?\\s+entonces\\s+|,?\\s+y\\s+tambi[e\xc3\xa9]n\\s+|,?\\s+adem[a\xc3\xa1]s\\s+|,?\\s+y\\s+por\\s+separado\\s+"
 "|,\\s+(?=\\w)|\\.\\s+(?=[A-Z])";
static const std::regex COORD_AND ("(" + COORD_BASE + "|,?\\s+and\\s+|,?\\s+y\\s+|,?\\s+e\\s+)", std::regex::icase);
static const std::regex COORD_NOAND("(" + COORD_BASE + ")", std::regex::icase);
static const std::regex LEAD_CONJ("^(?:and|also|plus|then|next|finally|y|e|tambien|luego|despues|entonces|ademas)\\s+", std::regex::icase);
static const std::regex HAS_WORD3("\\b\\w{3,}\\b");
// size gate: a sub-task likely to generate SUBSTANTIAL output (long generation) — fan-out only pays then
static const std::regex SUBST_LONG("\\b(explain|describe|analyze|analyse|essay|paragraph|report|story|article|analysis|function|class|script|program|implement|tutorial|guide|outline|derive|prove|elaborate|summariz|summaris|design|develop|draft|compose|detailed|thorough|comprehensive|in[ -]depth|step[ -]by[ -]step|escribe|explica|analiza|implementa|ensayo|desarroll|redact|informe|historia)\\b", std::regex::icase);
static const std::regex SUBST_SHORT("\\b(haiku|joke|tagline|slogan|one[ -]liner|rhyme|limerick|couplet|capital of|weather|how much|how many|whats|what is|name an?|tell me a|chiste|nombra)\\b", std::regex::icase);
static bool is_substantial(const std::string & c){ return std::regex_search(c, SUBST_LONG) && !std::regex_search(c, SUBST_SHORT); }
static const std::set<std::string> STOP = {"a","an","the","of","to","in","on","for","and","or","is","are","be","it","its","this","that","with","into","your","you","me","my","our","their","them","he","she","they","his","her","as","at","by","from","el","la","los","las","un","una","unos","unas","de","del","y","e","o","que","con","en","para","por","su","sus","lo","mi","tu"};

static int content_words(const std::string & s) {
    std::regex w("[A-Za-z][A-Za-z'\\-]*"); int n = 0;
    for (auto it = std::sregex_iterator(s.begin(), s.end(), w); it != std::sregex_iterator(); ++it) {
        std::string t = it->str(); for (auto & c : t) c = tolower(c);
        if (t.size() > 1 && !STOP.count(t)) n++;
    }
    return n;
}
static std::string trim(const std::string & s){ size_t a=s.find_first_not_of(" \t\n"); if(a==std::string::npos) return ""; size_t b=s.find_last_not_of(" \t\n"); return s.substr(a,b-a+1); }

static std::vector<std::string> split_clauses(const std::string & prompt) {
    // 1) protect quoted spans
    std::vector<std::string> quotes; std::string p; p.reserve(prompt.size());
    { std::regex q("\"[^\"]*\"|'[^']*'"); auto last = prompt.cbegin();
      for (auto it = std::sregex_iterator(prompt.begin(), prompt.end(), q); it != std::sregex_iterator(); ++it) {
          p.append(last, prompt.cbegin()+it->position()); p += "\x01" + std::to_string(quotes.size()) + "\x01";
          quotes.push_back(it->str()); last = prompt.cbegin()+it->position()+it->length(); }
      p.append(last, prompt.cend()); }

    std::vector<std::string> raw;
    int nlist = std::distance(std::sregex_iterator(p.begin(), p.end(), NUMLIST), std::sregex_iterator());
    if (nlist >= 2) {
        for (auto it = std::sregex_token_iterator(p.begin(), p.end(), NUMLIST, -1); it != std::sregex_token_iterator(); ++it)
            if (!trim(*it).empty()) raw.push_back(trim(*it));
    } else {
        int nverbs = std::distance(std::sregex_iterator(p.begin(), p.end(), TASK_VERB), std::sregex_iterator());
        bool has_comma = p.find(',') != std::string::npos;
        const std::regex & rx = (nverbs >= 2 || has_comma) ? COORD_AND : COORD_NOAND;
        for (auto it = std::sregex_token_iterator(p.begin(), p.end(), rx, -1); it != std::sregex_token_iterator(); ++it) {
            std::string c = trim(*it); if (!c.empty()) raw.push_back(c);
        }
    }
    // restore quotes + strip leading conjunction
    std::vector<std::string> clauses;
    for (auto & c : raw) {
        std::string r; size_t i = 0;
        while (i < c.size()) { if (c[i]=='\x01'){ size_t e=c.find('\x01', i+1); int idx=std::stoi(c.substr(i+1,e-i-1)); r+=quotes[idx]; i=e+1; } else r+=c[i++]; }
        r = trim(std::regex_replace(r, LEAD_CONJ, ""));
        clauses.push_back(r);
    }
    // merge an ISOLATED short fragment (<=1 content word, no verb)
    std::vector<int> shorts;
    for (size_t i = 0; i < clauses.size(); i++)
        if (!std::regex_search(clauses[i], TASK_VERB) && content_words(clauses[i]) <= 1) shorts.push_back(i);
    if (shorts.size() == 1 && shorts[0] > 0) {
        int i = shorts[0]; std::string m = clauses[i-1]; if(!m.empty()&&m.back()=='.') m.pop_back();
        clauses[i-1] = m + ", " + clauses[i]; clauses.erase(clauses.begin()+i);
    }
    std::vector<std::string> out;
    for (auto & c : clauses) if (std::regex_search(c, HAS_WORD3)) out.push_back(c);
    return out;
}

// ---------------- llama embedding ([CLS], raw) ----------------
static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static int g_n_ctx = 512;
static std::vector<float> embed_one(const std::string & text) {
    llama_memory_clear(llama_get_memory(g_ctx), true);
    std::vector<llama_token> toks = common_tokenize(g_ctx, text, true, true);
    if ((int)toks.size() > g_n_ctx) toks.resize(g_n_ctx);
    llama_batch batch = llama_batch_init((int)toks.size(), 0, 1);
    for (int i = 0; i < (int)toks.size(); i++) common_batch_add(batch, toks[i], i, {0}, true);
    std::vector<float> out(NE, 0.0f);
    if (llama_decode(g_ctx, batch) == 0) {
        const float * e = llama_get_embeddings_seq(g_ctx, 0);
        if (e) common_embd_normalize(e, out.data(), NE, -1);    // -1 = no normalization (raw)
    }
    llama_batch_free(batch);
    return out;
}

// ---------------- decompose ----------------
static std::string json_escape(const std::string & s){ std::string o; for(char c:s){ if(c=='"'||c=='\\')o+='\\',o+=c; else if(c=='\n')o+="\\n"; else if((unsigned char)c<0x20)continue; else o+=c; } return o; }

static std::string decompose_json(const std::string & prompt) {
    std::vector<std::string> cl = split_clauses(prompt);
    int n = (int)cl.size();
    auto build = [&](const std::vector<std::vector<int>> & deps, bool fanout, bool recommend, int width, const char * klass){
        std::string j = "{\"subtasks\":[";
        for (int i = 0; i < (int)cl.size(); i++) {
            j += std::string(i?",":"") + "{\"id\":\"" + std::string(1,(char)('a'+i)) + "\",\"text\":\"" + json_escape(cl[i]) + "\",\"deps\":[";
            for (size_t k = 0; k < deps[i].size(); k++) j += std::string(k?",":"") + "\"" + std::string(1,(char)('a'+deps[i][k])) + "\"";
            j += "]}";
        }
        j += "],\"cls\":\"" + std::string(klass) + "\",\"fanout\":" + (fanout?"true":"false") + ",\"maxWidth\":" + std::to_string(width) + ",\"recommend\":" + (recommend?"true":"false") + "}";
        return j;
    };
    if (n < 2) { std::vector<std::vector<int>> d(std::max(n,1)); if(n==0){cl.push_back(prompt); d.resize(1);} return build(d, false, false, 1, "atomic"); }
    std::vector<std::vector<int>> deps(n);
    for (int i = 0; i < n; i++) for (int j = i+1; j < n; j++) {
        std::string x = prompt + " [SEP] Does this step depend on the earlier step? EARLIER: " + cl[i] + " LATER: " + cl[j];
        if (head_predict(embed_one(x).data())) deps[j].push_back(i);
    }
    // topo levels -> max width
    std::set<int> done; int width = 0, guard = 0; bool has_edge = false;
    for (auto & d : deps) if (!d.empty()) has_edge = true;
    while ((int)done.size() < n && guard++ < 40) {
        std::vector<int> ready;
        for (int k = 0; k < n; k++) if (!done.count(k)) { bool ok=true; for(int d:deps[k]) if(!done.count(d)){ok=false;break;} if(ok) ready.push_back(k); }
        if (ready.empty()) { for(int k=0;k<n;k++) if(!done.count(k)) ready.push_back(k); }
        width = std::max(width, (int)ready.size()); for(int k:ready) done.insert(k);
    }
    bool fanout = width >= 2;
    const char * klass = !has_edge ? "parallel" : (width == 1 ? "dependent" : "mixed");
    int n_indep_sub = 0; for (int i = 0; i < n; i++) if (deps[i].empty() && is_substantial(cl[i])) n_indep_sub++;
    bool recommend = fanout && n_indep_sub >= 2;
    return build(deps, fanout, recommend, width, klass);
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) return 1;
    params.embedding = true;
    if (params.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    if (params.n_ctx == 0) params.n_ctx = 512;
    params.n_batch = std::max<int>(params.n_batch, params.n_ctx);
    params.n_ubatch = params.n_batch;

    const char * head_path = getenv("HEAD"); if (!head_path) head_path = "microme-head.bin";
    int port = getenv("PORT") ? atoi(getenv("PORT")) : 8099;

    llama_backend_init(); llama_numa_init(params.numa);
    static auto llama_init = common_init_from_params(params);
    g_model = llama_init->model(); g_ctx = llama_init->context();
    if (!g_model) { fprintf(stderr, "model load failed\n"); return 1; }
    g_n_ctx = llama_n_ctx(g_ctx);
    if (!load_head(head_path)) { fprintf(stderr, "head load failed: %s\n", head_path); return 1; }
    fprintf(stderr, "decompose-server: model+head loaded, n_ctx=%d, listening on :%d\n", g_n_ctx, port);

    httplib::Server svr;
    svr.Get("/", [](const httplib::Request&, httplib::Response& res){ res.set_content("{\"status\":\"ok\"}", "application/json"); });
    svr.Post("/decompose", [](const httplib::Request& req, httplib::Response& res){
        // minimal JSON: extract "prompt" value
        std::string b = req.body, prompt; size_t k = b.find("\"prompt\"");
        if (k != std::string::npos) { size_t colon = b.find(':', k+8); size_t q = b.find('"', colon+1);
            for (size_t i = q+1; i < b.size(); i++) { if (b[i]=='\\' && i+1<b.size()) { char c=b[i+1]; prompt += (c=='n'?'\n':c); i++; } else if (b[i]=='"') break; else prompt += b[i]; } }
        res.set_content(decompose_json(prompt), "application/json");
    });
    svr.listen("0.0.0.0", port);
    return 0;
}
