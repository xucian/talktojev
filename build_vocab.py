"""
Build vocab tree v3.
- Break up the 'thing' catch-all into proper sub-categories
- Max 25 children per node (empirically proven Jev handles this fine)
- Output as folder structure: vocab/<category>.json + vocab/_index.json
- Keyword-rich labels at every level
"""

import json
import os
import nltk

HERE = os.path.dirname(os.path.abspath(__file__))
WORDS_PATH = os.path.join(HERE, 'words.json')
OUTPUT_DIR = os.path.join(HERE, 'vocab')
MAX_CHILDREN = 200

with open(WORDS_PATH) as f:
    ALL_WORDS = json.load(f)
WORD_SET = set(ALL_WORDS)

tagged = nltk.pos_tag(ALL_WORDS)
WORD_POS = {w: tag for w, tag in tagged}

CATEGORIES = [
    {
        "id": "social",
        "label": "a social word — hello, hi, thanks, please, sorry, yes, no, okay, sure, agree",
        "seeds": {"hello", "hi", "hey", "welcome", "goodbye", "bye", "thanks",
                  "thank", "thanked", "please", "sorry", "excuse", "pardon",
                  "cheers", "greetings", "ok", "okay", "yes", "no", "yeah",
                  "nope", "sure", "alright", "indeed", "absolutely", "exactly",
                  "certainly", "definitely", "totally", "agree", "disagree",
                  "correct", "true", "false", "right", "wrong", "never",
                  "neither", "nor", "none", "refuse", "reject",
                  "oh", "wow", "gonna", "gotta", "wanna",
                  "damn", "hell", "ass", "miss", "mom"},
    },
    {
        "id": "pronoun",
        "label": "a person reference — I, you, he, she, it, we, they, me, my, your, who, someone",
        "seeds": {"i", "me", "my", "myself", "mine", "you", "your", "yourself",
                  "yours", "he", "him", "his", "himself", "she", "her", "hers",
                  "herself", "it", "its", "itself", "we", "us", "our", "ours",
                  "ourselves", "they", "them", "their", "theirs", "themselves",
                  "who", "whom", "whose", "what", "which",
                  "someone", "something", "anyone", "anything",
                  "everyone", "everything", "nobody", "nothing", "somebody",
                  "each", "other", "another", "both", "either",
                  "one", "ones", "own", "self", "such", "whatever", "whoever",
                  "wherever", "whenever", "however", "whichever"},
        "pos_match": {"PRP", "PRP$", "WP", "WP$"},
    },
    {
        "id": "grammar",
        "label": "a connecting word — is, am, are, was, the, a, to, for, and, but, or, not, can, will",
        "seeds": {"the", "a", "an", "this", "that", "these", "those",
                  "and", "but", "or", "so", "because", "if", "though",
                  "although", "however", "therefore", "thus", "since",
                  "unless", "whether", "while", "whereas", "besides",
                  "moreover", "also", "too", "yet", "still", "instead",
                  "otherwise", "anyway", "meanwhile", "nevertheless",
                  "nonetheless", "regardless", "despite", "except",
                  "rather", "than",
                  "to", "in", "of", "for", "with", "on", "at", "about",
                  "from", "by", "into", "out", "up", "down", "off",
                  "over", "under", "through", "between", "among",
                  "against", "within", "without", "toward", "towards",
                  "upon", "along", "across", "around", "behind",
                  "beyond", "during", "until", "before", "after",
                  "above", "below", "near", "beside", "beneath",
                  "throughout", "per", "via", "plus", "as", "like",
                  "is", "am", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "having",
                  "do", "does", "did", "doing", "done",
                  "can", "could", "would", "should", "may", "might",
                  "shall", "will", "must", "need",
                  "not", "dont", "doesnt", "didnt", "cant", "wont",
                  "shouldnt", "wouldnt", "couldnt", "isnt", "arent",
                  "wasnt", "werent", "havent", "hasnt", "hadnt"},
        "pos_match": {"CC", "IN", "DT", "TO", "RP", "MD", "EX"},
    },
    {
        "id": "feeling",
        "label": "a feeling or desire — good, happy, sad, love, hate, want, hope, care, enjoy, fine, well",
        "seeds": {"good", "great", "happy", "fine", "well", "wonderful", "awesome",
                  "nice", "excellent", "amazing", "fantastic", "beautiful",
                  "love", "lovely", "glad", "pleased", "proud", "excited",
                  "grateful", "comfortable", "confident", "calm", "peaceful",
                  "safe", "brilliant", "perfect", "incredible", "cheerful",
                  "content", "satisfied", "relieved", "hopeful",
                  "fun", "funny", "interesting", "enjoyable", "pleasant",
                  "sweet", "kind", "gentle", "friendly", "caring",
                  "dear", "precious", "favorite", "best", "better",
                  "bad", "sorry", "sad", "angry", "upset", "hurt", "sick",
                  "tired", "worried", "afraid", "scared", "nervous", "anxious",
                  "confused", "lost", "alone", "lonely", "bored", "frustrated",
                  "disappointed", "embarrassed", "ashamed", "guilty",
                  "terrible", "horrible", "awful", "ugly", "painful", "weak",
                  "poor", "worse", "worst", "broken", "dangerous", "difficult",
                  "hard", "tough", "cruel", "mean", "evil", "strange",
                  "weird", "odd", "unfortunate",
                  "feel", "feeling", "felt", "emotion", "mood",
                  "want", "wanted", "hope", "wish", "prefer",
                  "enjoy", "enjoyed", "hate", "hated",
                  "fear", "doubt", "trust", "worry", "care", "mind",
                  "surprise", "surprised", "shock", "impressed",
                  "cool", "warm"},
    },
    {
        "id": "think",
        "label": "a thinking word — think, know, believe, understand, learn, remember, see, hear, idea",
        "seeds": {"think", "know", "believe", "understand", "learn", "remember",
                  "forget", "realize", "recognize", "discover", "notice",
                  "imagine", "wonder", "guess", "assume", "suppose", "expect",
                  "consider", "decide", "plan", "compare", "judge",
                  "analyze", "study", "research", "examine", "focus",
                  "observe", "watch", "look", "see", "seen", "saw",
                  "hear", "heard", "listen", "sense", "experience",
                  "dream", "reflect", "remind", "recall", "review",
                  "figure", "solve", "reason", "conclude", "determine",
                  "identify", "define", "interpret", "predict", "prove",
                  "confirm", "verify", "accept", "approve", "allow",
                  "idea", "thought", "thinking", "knowledge", "understanding",
                  "belief", "opinion", "view", "perspective", "theory",
                  "concept", "principle", "logic", "intelligence",
                  "wisdom", "awareness", "consciousness", "attention",
                  "memory", "imagination", "creativity", "curiosity",
                  "insight", "judgment", "perception", "recognition",
                  "considered", "knowing", "known",
                  "found", "finding", "aware", "based"},
    },
    {
        "id": "communicate",
        "label": "a communication word — say, tell, ask, talk, speak, write, read, call, explain, story",
        "seeds": {"say", "said", "tell", "told", "ask", "asked", "talk",
                  "talked", "speak", "spoke", "spoken", "call", "called",
                  "write", "wrote", "written", "read", "name", "named",
                  "describe", "explain", "discuss", "suggest", "recommend",
                  "warn", "promise", "claim", "state", "announce", "declare",
                  "report", "mention", "refer", "express", "respond",
                  "reply", "comment", "complain", "praise", "apologize",
                  "invite", "request", "demand", "order", "command",
                  "instruct", "teach", "guide", "direct", "point",
                  "indicate", "signal", "imply", "hint", "whisper",
                  "shout", "cry", "laugh", "smile", "joke",
                  "word", "words", "language", "speech", "voice",
                  "conversation", "message", "question", "answer",
                  "story", "stories", "song", "poem", "quote",
                  "writing", "reading", "note", "meaning", "meant",
                  "communication", "expression", "statement", "description",
                  "explanation", "argument", "debate", "discussion",
                  "information", "news", "media", "press",
                  "advertising", "announcement", "cable", "chart",
                  "clip", "documents", "feedback", "footage",
                  "interviews", "publications", "script", "texts",
                  "thread", "warnings"},
    },
    {
        "id": "action",
        "label": "an action — go, come, take, give, make, get, put, move, do, run, work, play, help",
        "seeds": {"go", "going", "went", "gone", "come", "came", "coming",
                  "take", "took", "taken", "taking", "give", "gave", "given",
                  "giving", "make", "made", "making", "get", "got", "getting",
                  "put", "putting", "set", "setting", "run", "running", "ran",
                  "move", "moved", "turn", "turned", "pull", "push",
                  "hold", "held", "keep", "kept", "let", "leave", "left",
                  "bring", "brought", "send", "sent", "carry", "catch",
                  "caught", "throw", "drop", "pick", "picked",
                  "cut", "break", "broke", "build", "built", "create",
                  "created", "open", "opened", "close", "closed",
                  "start", "started", "stop", "stopped", "begin", "began",
                  "end", "ended", "finish", "finished",
                  "continue", "follow", "followed", "lead", "led",
                  "drive", "drove", "walk", "walked", "sit", "sat",
                  "stand", "stood", "fall", "fell", "rise", "rose",
                  "grow", "grew", "grown", "change", "changed",
                  "add", "added", "remove", "removed",
                  "play", "played", "work", "worked", "use", "used",
                  "try", "tried", "act", "serve", "served",
                  "provide", "provided", "offer", "offered",
                  "share", "shared", "pass", "passed", "reach", "reached",
                  "enter", "entered", "return", "returned",
                  "meet", "met", "join", "joined", "connect",
                  "fix", "fixed", "handle", "managed",
                  "spend", "spent", "pay", "paid", "buy", "bought",
                  "sell", "sold", "win", "won", "lose", "fight",
                  "attack", "defend", "protect", "support", "help",
                  "helped", "save", "saved", "happen", "happened",
                  "live", "lived", "die", "died", "kill", "killed",
                  "eat", "sleep", "wear", "wore", "worn",
                  "hit", "shot", "hang", "raise", "raised",
                  "draw", "drawn", "spread", "sing", "sang",
                  "strike", "stuck", "blow", "shut",
                  "born", "become", "became", "appear", "appeared",
                  "remain", "stayed", "exist", "involved",
                  "require", "required", "include", "included",
                  "contain", "involve",
                  "result", "resulted", "cause", "caused",
                  "affect", "affected", "produce", "produced",
                  "develop", "developed", "improve", "improved",
                  "increase", "increased", "reduce", "reduced",
                  "apply", "applied", "cover", "covered",
                  "establish", "established", "form", "formed",
                  "allow", "allowed",
                  "check", "checked", "stay", "playing", "works", "looks",
                  "post", "posted", "needs", "watch", "watching",
                  "rest", "rested", "hang", "hanging",
                  "dance", "swim", "jump", "climb", "fly", "flew",
                  "search", "searched", "clean", "cleaned",
                  "cook", "cooked", "wash", "washed",
                  "acts", "applies", "becomes", "begins", "belongs",
                  "breaks", "brings", "calls", "celebrate", "commit",
                  "convince", "deny", "deserve", "deserves", "destroy",
                  "encourage", "explains", "fails", "falls", "finds",
                  "fits", "gather", "grab", "ignore", "investigate",
                  "keeps", "lift", "loads", "manage", "meets",
                  "opens", "overcome", "pays", "perform",
                  "picks", "prevent", "promote", "reads", "runs",
                  "sees", "seek", "shake", "shoot", "steal",
                  "stops", "tells", "tends", "treat", "tries",
                  "trigger", "visits", "wake", "walks", "writes"},
    },
    {
        "id": "describe",
        "label": "a describing word — big, small, new, old, different, important, same, long, high, fast",
        "seeds": {"big", "small", "large", "little", "long", "short", "tall",
                  "high", "low", "wide", "deep", "thick", "thin",
                  "heavy", "light", "fast", "slow", "quick",
                  "new", "old", "young", "modern", "recent", "current",
                  "different", "similar", "same", "equal", "various",
                  "special", "specific", "particular", "general", "common",
                  "usual", "normal", "typical", "standard", "regular",
                  "ordinary", "simple", "easy", "complex", "clear",
                  "certain", "obvious", "single", "extra", "separate",
                  "independent", "unique", "rare", "original",
                  "popular", "famous", "real", "actual", "genuine",
                  "natural", "physical", "mental", "social",
                  "political", "economic", "financial", "legal",
                  "official", "public", "private", "personal",
                  "professional", "technical", "scientific",
                  "international", "national", "local", "global",
                  "available", "ready", "free", "full", "empty",
                  "rich", "expensive", "cheap", "valuable",
                  "effective", "successful", "powerful", "strong",
                  "solid", "stable", "secure", "clean", "fresh",
                  "dry", "wet", "hot", "cold", "dark", "bright",
                  "sharp", "smooth", "soft", "round", "flat",
                  "straight", "wild", "raw", "pure", "whole",
                  "complete", "total", "entire", "absolute",
                  "direct", "correct", "proper", "appropriate",
                  "fair", "reasonable", "responsible", "careful",
                  "alive", "active", "busy", "quiet", "silent",
                  "main", "major", "minor", "important", "significant",
                  "serious", "critical", "key", "central", "primary",
                  "basic", "fundamental", "essential", "necessary",
                  "traditional", "classic", "religious", "cultural",
                  "military", "medical", "educational", "environmental",
                  "digital",
                  "likely", "unlikely", "possible", "impossible",
                  "able", "unable", "capable", "positive", "negative",
                  "external", "internal", "upper", "lower",
                  "male", "female", "human", "ancient",
                  "rapid", "severe", "rural", "urban"},
    },
    {
        "id": "time_manner",
        "label": "a time or manner word — now, then, always, very, really, just, also, only, too, how, why",
        "seeds": {"now", "then", "today", "tomorrow", "yesterday", "tonight",
                  "always", "never", "sometimes", "often", "usually",
                  "already", "still", "yet", "soon", "recently",
                  "finally", "eventually", "immediately", "suddenly",
                  "once", "again", "ago", "ever", "forever",
                  "very", "really", "just", "also", "only", "too",
                  "quite", "rather", "pretty", "extremely",
                  "completely", "totally", "entirely", "fully",
                  "exactly", "merely", "simply", "hardly", "barely",
                  "nearly", "almost", "slightly", "somewhat", "fairly",
                  "particularly", "especially", "generally", "mostly",
                  "mainly", "primarily", "essentially", "basically",
                  "certainly", "definitely", "surely", "obviously",
                  "clearly", "apparently", "probably", "possibly",
                  "perhaps", "maybe", "seriously", "honestly",
                  "personally", "technically", "officially",
                  "together", "alone", "apart", "away", "ahead",
                  "forward", "back", "even", "else", "enough",
                  "somehow", "otherwise",
                  "twice", "daily", "early", "late", "about",
                  "how", "why", "when", "where"},
        "pos_match": {"RB", "RBR", "RBS", "WRB"},
    },
    {
        "id": "number",
        "label": "a number or quantity — one, two, many, some, all, every, more, most, first, last",
        "seeds": {"one", "two", "three", "four", "five", "six", "seven",
                  "eight", "nine", "ten", "hundred", "thousand", "million",
                  "billion", "half", "quarter", "dozen", "pair",
                  "some", "many", "much", "few", "several", "all", "any",
                  "every", "each", "both", "more", "most", "less", "least",
                  "enough", "plenty", "lot", "lots", "number", "amount",
                  "total", "average", "maximum", "minimum", "double",
                  "percent", "rate", "degree", "score",
                  "first", "second", "third", "next", "last"},
        "pos_match": {"CD"},
    },
    {
        "id": "people",
        "label": "a person or group — person, people, man, woman, friend, family, child, team, company",
        "seeds": {"person", "people", "man", "men", "woman", "women",
                  "child", "children", "baby", "boy", "girl", "kid", "kids",
                  "adult", "friend", "friends", "family", "families",
                  "father", "mother", "parent", "parents", "son", "daughter",
                  "brother", "sister", "husband", "wife", "partner",
                  "neighbor", "stranger", "guest", "customer", "client",
                  "patient", "student", "students", "teacher", "professor",
                  "doctor", "nurse", "lawyer", "judge", "officer",
                  "soldier", "king", "queen", "president", "leader",
                  "manager", "boss", "worker", "workers", "employee",
                  "staff", "team", "teams", "group", "groups",
                  "member", "members", "community", "society",
                  "population", "citizen", "citizens",
                  "audience", "crowd", "individual", "individuals",
                  "being", "character", "hero",
                  "artist", "writer", "author", "player", "players",
                  "coach", "expert", "master", "mr", "mrs", "sir",
                  "lady", "lord", "god", "spirit", "soul",
                  "victim", "enemy", "couple",
                  "organization", "company", "companies",
                  "government", "governments", "party", "parties",
                  "nation", "nations", "army",
                  "christian", "muslim", "jewish",
                  "guy", "guys", "police", "fan", "fans", "volunteer",
                  "candidate", "champion", "opponent", "rival", "spy",
                  "prince", "princess", "mayor", "governor", "minister",
                  "ambassador", "spokesman", "representative",
                  "agents", "athletes", "brothers", "boyfriend",
                  "chairman", "champions", "chef", "clients", "coaches",
                  "colleagues", "commissioner", "consumers", "critics",
                  "customers", "dad", "daddy", "daughters", "deputy",
                  "developers", "doctors", "emperor", "engineers",
                  "experts", "faculty", "females", "fighter", "folk", "folks",
                  "founder", "ghost", "giants", "gods", "grandfather",
                  "guards", "guests", "heroes", "homeless", "hosts",
                  "humans", "hunter", "immigrants", "killer",
                  "ladies", "makers", "mama", "manufacturers", "masters",
                  "ministers", "mothers", "opponents", "operators",
                  "owners", "participants", "partners", "peoples",
                  "persons", "personnel", "pet", "pilot", "pope",
                  "premier", "priest", "producers", "professionals",
                  "researchers", "residents", "saints", "scientists",
                  "secretary", "singles", "sisters", "sons", "speakers",
                  "supporters", "twins", "veterans", "victims", "viewers",
                  "visitors", "volunteers", "warriors", "winners", "writers"},
    },
    {
        "id": "world",
        "label": "a place or thing in the world — home, world, country, city, room, body, animal, nature",
        "seeds": {"place", "home", "house", "room", "building", "office",
                  "school", "schools", "hospital", "church", "store", "shop",
                  "restaurant", "hotel", "station", "park", "garden",
                  "street", "road", "path", "bridge", "wall", "door",
                  "window", "floor", "ground", "roof", "top", "bottom",
                  "side", "front", "center", "middle", "edge", "corner",
                  "inside", "outside", "here", "there",
                  "everywhere", "nowhere", "somewhere", "anywhere",
                  "area", "region", "space", "field",
                  "land", "country", "countries", "state", "states",
                  "city", "cities", "town", "village",
                  "world", "earth", "planet", "universe",
                  "north", "south", "east", "west",
                  "position", "direction", "location", "distance",
                  "scene", "setting", "environment", "market",
                  "court", "base", "site", "campus",
                  "surface", "border", "line",
                  "body", "head", "face", "eye", "eyes", "ear",
                  "nose", "mouth", "hand", "hands", "arm", "arms",
                  "leg", "legs", "foot", "feet", "heart", "blood",
                  "skin", "hair", "bone", "brain", "cell", "cells",
                  "finger", "shoulder", "chest", "stomach", "neck",
                  "muscle", "lip", "tooth", "tongue", "throat",
                  "animal", "animals", "dog", "cat", "bird", "fish",
                  "horse", "tree", "trees", "flower", "plant", "plants",
                  "forest", "mountain", "river", "sea", "ocean",
                  "island", "sun", "moon", "star", "stars",
                  "fire", "air", "wind", "rain", "snow", "weather",
                  "water", "ice", "nature", "life", "death", "birth",
                  "health", "disease", "food", "meal",
                  "spring", "summer", "winter", "fall",
                  "college", "university", "airport", "farm",
                  "camp", "library", "museum", "theater", "stadium",
                  "temple", "castle", "palace", "tower", "port",
                  "beach", "desert", "valley", "hill", "cave",
                  "lake", "pond", "stream", "creek", "bay",
                  "jungle", "wilderness", "marsh", "cliff"},
    },
    # --- FORMER 'thing' CATCH-ALL, NOW SPLIT INTO SUB-CATEGORIES ---
    {
        "id": "concept",
        "label": "an idea or concept — way, idea, fact, reason, problem, plan, system, goal, purpose",
        "seeds": {"way", "ways", "part", "parts", "kind", "type", "form",
                  "forms", "point", "points", "reason", "reasons",
                  "fact", "facts", "truth", "reality", "theory",
                  "concept", "principle", "rule", "rules", "law", "laws",
                  "standard", "standards", "policy", "method", "approach",
                  "strategy", "plan", "plans", "system", "systems",
                  "process", "step", "steps", "stage",
                  "model", "pattern", "structure", "framework",
                  "source", "cause", "effect", "result", "results",
                  "outcome", "impact", "benefit", "advantage",
                  "opportunity", "option", "choice", "decision",
                  "solution", "answer", "response", "reaction",
                  "effort", "attempt", "success", "achievement",
                  "progress", "development", "growth",
                  "difference", "differences", "connection", "relationship",
                  "example", "examples", "version", "series",
                  "category", "class", "section", "chapter",
                  "topic", "subject", "issue", "issues", "matter",
                  "problem", "problems", "challenge", "task",
                  "goal", "goals", "purpose", "aim", "mission",
                  "value", "values", "quality", "feature", "aspect",
                  "factor", "detail", "details", "condition", "conditions",
                  "context", "background", "tradition", "culture",
                  "style", "fashion", "trend", "movement",
                  "innovation", "technology", "science",
                  "meaning", "sense", "power", "force", "strength",
                  "ability", "skill", "talent", "experience",
                  "education", "training", "practice", "test",
                  "research", "analysis", "record", "records",
                  "account", "project", "projects", "program", "programs",
                  "service", "services", "product", "products",
                  "action", "activity", "activities", "performance",
                  "advice", "focus", "access", "range",
                  "level", "extent", "basis", "term", "terms",
                  "list", "figure", "view",
                  "possibility", "potential", "limit", "limits",
                  "influence", "mark", "signal", "code",
                  "address", "reference", "guide", "instruction",
                  "suggestion", "recommendation",
                  "pressure", "stress", "energy", "material",
                  "network", "production", "operation", "operations",
                  "addition", "article", "chance", "changes", "charge",
                  "course", "evidence", "release", "sort", "size",
                  "track", "text", "cross", "capital", "questions",
                  "sales", "areas", "places", "others", "lives",
                  "association", "union", "club", "district",
                  "aspects", "balance", "beliefs", "burden",
                  "capacity", "challenges", "chances", "characteristics",
                  "choices", "circumstances", "claims", "combination",
                  "comparison", "complaints", "components", "concerns",
                  "conduct", "connections", "consequences", "controversy",
                  "core", "coverage", "damage", "deaths", "decades",
                  "decisions", "default", "delay", "desire", "developments",
                  "directions", "discussions", "efforts", "elements",
                  "emotions", "error", "errors", "estimates", "exception",
                  "exchange", "existence", "expectations", "exposure",
                  "factors", "failure", "fame", "fate", "fault", "favor",
                  "favourite", "features", "findings", "fortune", "functions",
                  "gain", "gains", "gap", "glory", "grounds", "guarantee",
                  "guidelines", "habit", "happiness", "highlights",
                  "humanity", "ideas", "identity", "images",
                  "importance", "impression", "improvements", "index",
                  "infrastructure", "input", "inspiration", "instance",
                  "instructions", "integrity", "intent", "intention",
                  "interaction", "interests", "introduction",
                  "investigation", "involvement", "joy", "kinds", "kingdom",
                  "lack", "landscape", "languages", "layer", "levels",
                  "license", "locations", "loop", "luxury",
                  "majority", "manner", "margin", "matters", "measures",
                  "mechanism", "membership", "memories", "mess", "methods",
                  "minority", "mistakes", "mix", "mode", "models",
                  "moments", "motion", "movements", "objects",
                  "occasion", "occasions", "offers", "opinions",
                  "opportunities", "opposition", "options", "orders",
                  "output", "ownership", "pace", "panel", "panic",
                  "participation", "partnership", "passage", "passion",
                  "percentage", "performances", "periods", "permission",
                  "personality", "philosophy", "pieces", "planning",
                  "popularity", "portion", "possession", "positions",
                  "powers", "practices", "preparation", "presence",
                  "prevention", "pride", "principles", "priority",
                  "privacy", "processes", "promises", "properties",
                  "protest", "psychology", "purposes", "reactions",
                  "recommendations", "references", "regard", "regions",
                  "registration", "regret", "relations", "relationships",
                  "religion", "removal", "replacement", "reports",
                  "reputation", "requirements", "reserves", "residence",
                  "resistance", "resort", "responses", "restrictions",
                  "returns", "reviews", "reward", "risks", "romance",
                  "rounds", "sacrifice", "savings", "scenes", "scheme",
                  "scope", "scores", "secrets", "sections", "sector",
                  "selection", "sequence", "sessions", "settings",
                  "settlement", "shelter", "sight", "signature",
                  "silence", "sin", "situations", "skills", "solutions",
                  "sources", "spaces", "stability", "stages",
                  "statements", "statistics", "status", "strategies",
                  "structures", "struggle", "studies", "styles",
                  "suggestions", "supplies", "supports", "survival",
                  "targets", "techniques", "technologies", "tension",
                  "territory", "theories", "thoughts", "threats",
                  "ties", "tips", "titles", "tools", "topics",
                  "tracks", "transactions", "transfer", "transition",
                  "trends", "tribute", "trouble", "types",
                  "unity", "usage", "utility", "variety", "venture",
                  "venue", "versions", "vision", "voices",
                  "warning", "worship", "zero"},
    },
    {
        "id": "society",
        "label": "a society or work word — business, money, job, career, economy, market, trade, law, role",
        "seeds": {"business", "industry", "market", "economy",
                  "trade", "investment", "profit", "loss",
                  "debt", "budget", "fund", "funds", "resource",
                  "resources", "supply", "demand", "interest",
                  "share", "shares", "property", "rights",
                  "freedom", "justice", "peace", "security", "safety",
                  "risk", "threat", "control", "management", "leadership",
                  "role", "roles", "job", "jobs", "career", "position",
                  "responsibility", "duty", "agreement", "contract",
                  "deal", "proposal",
                  "support", "attention", "order",
                  "campaign", "debate", "discussion",
                  "election", "vote", "democracy", "reform",
                  "crisis", "conflict", "violence", "crime",
                  "prison", "sentence", "trial", "justice",
                  "tax", "taxes", "income", "salary", "wage",
                  "payment", "credit", "loan", "insurance",
                  "pension", "stock", "bond", "asset",
                  "estate", "wealth", "poverty",
                  "war", "wars", "battle", "peace",
                  "aid", "relief", "welfare", "charity",
                  "commission", "committee", "council", "board",
                  "department", "ministry", "agency", "bureau",
                  "authority", "administration", "institution",
                  "regulation", "legislation", "amendment",
                  "constitution", "treaty", "resolution",
                  "revolution", "independence", "liberty",
                  "agencies", "boards", "communities", "contracts",
                  "controls", "cooperation", "corruption", "deals",
                  "discrimination", "diversity", "duties", "economics",
                  "equality", "immigration", "labor", "organisations",
                  "organizations", "ownership", "partnership",
                  "participation", "racism", "ranks", "securities",
                  "unions", "welfare"},
    },
    {
        "id": "event_time",
        "label": "a time or event — time, day, year, moment, history, event, game, meeting, morning",
        "seeds": {"time", "times", "day", "days", "night", "nights",
                  "morning", "afternoon", "evening", "weekend",
                  "week", "weeks", "month", "months", "year", "years",
                  "decade", "century", "season", "moment", "minute",
                  "minutes", "hour", "hours", "date", "age",
                  "period", "era", "history", "past", "present", "future",
                  "event", "events", "news", "report",
                  "case", "cases", "situation",
                  "accident", "incident", "attack",
                  "game", "games", "match", "race",
                  "trip", "journey", "tour",
                  "meeting", "conference", "ceremony", "wedding",
                  "celebration", "festival", "holiday", "vacation",
                  "beginning", "ending",
                  "schedule", "deadline", "appointment"},
    },
    {
        "id": "object",
        "label": "an object or thing — thing, book, car, phone, tool, device, machine, food, clothes",
        "seeds": {"thing", "things", "stuff", "object", "item", "piece",
                  "bit", "tool", "device", "machine", "equipment",
                  "weapon", "gun", "knife", "sword",
                  "car", "cars", "bus", "train", "plane",
                  "ship", "boat", "truck", "vehicle",
                  "phone", "computer", "screen", "camera", "radio",
                  "television", "tv", "internet", "website", "email",
                  "software", "program", "data",
                  "metal", "stone", "glass", "plastic", "wood", "paper",
                  "card", "box", "bag", "bottle", "cup", "plate",
                  "table", "chair", "bed", "desk", "furniture",
                  "clock", "watch", "key", "lock", "light", "lamp",
                  "flag", "sign", "map", "book", "books", "page", "pages",
                  "letter", "letters", "document", "file", "picture",
                  "image", "photo", "photograph", "film", "movie", "movies",
                  "video", "show", "shows", "music", "art",
                  "painting", "design",
                  "money", "dollar", "dollars", "cash", "price", "cost",
                  "bill", "bills", "coin",
                  "food", "bread", "meat", "fruit", "rice",
                  "sugar", "salt", "oil", "tea", "coffee", "wine",
                  "beer", "milk", "egg", "eggs", "cake",
                  "chocolate", "drink", "clothes", "dress",
                  "shirt", "shoe", "shoes", "hat", "coat", "suit",
                  "belt", "ring", "medicine", "drug", "drugs",
                  "ball", "rock", "gold", "gas", "pain", "sex",
                  "bank", "banks", "miles", "visit", "george",
                  "cannot"},
    },
    {
        "id": "science",
        "label": "a science or tech word — science, technology, computer, research, data, digital, medical",
        "seeds": {"science", "technology", "computer", "engineering", "math",
                  "research", "experiment", "laboratory", "chemical", "chemistry",
                  "physics", "biology", "medicine", "medical", "surgery",
                  "therapy", "treatment", "virus", "infection", "vaccine",
                  "genetic", "evolution", "species", "organism", "bacteria",
                  "molecule", "atom", "electron", "nuclear", "radiation",
                  "energy", "electricity", "battery", "circuit", "chip",
                  "software", "hardware", "algorithm", "database", "server",
                  "internet", "digital", "virtual", "online", "cyber",
                  "robot", "satellite", "telescope", "microscope",
                  "ai", "intelligence", "artificial", "neural", "machine",
                  "sensor", "signal", "frequency", "wave", "spectrum",
                  "carbon", "oxygen", "hydrogen", "nitrogen", "iron",
                  "steel", "copper", "aluminum", "fuel", "solar",
                  "climate", "pollution", "emission", "waste", "recycle",
                  "temperature", "pressure", "gravity", "velocity", "mass",
                  "volume", "density", "formula", "equation", "variable",
                  "hypothesis", "conclusion", "observation", "measurement",
                  "laboratory", "specimen", "sample", "tissue", "cell",
                  "protein", "gene", "dna", "genome", "stem",
                  "diagnosis", "symptom", "dose", "prescription",
                  "clinical", "pharmaceutical", "antibiotic"},
    },
    {
        "id": "arts",
        "label": "an arts or entertainment word — art, music, film, movie, game, sport, song, story",
        "seeds": {"art", "arts", "music", "musical", "film", "films",
                  "movie", "movies", "show", "shows", "theater", "drama",
                  "comedy", "horror", "fiction", "novel", "literature",
                  "poetry", "painting", "sculpture", "photography",
                  "dance", "ballet", "opera", "concert", "festival",
                  "album", "song", "songs", "band", "guitar", "piano",
                  "drum", "instrument", "rhythm", "melody", "harmony",
                  "sport", "sports", "football", "basketball", "baseball",
                  "soccer", "tennis", "golf", "hockey", "boxing",
                  "swimming", "racing", "championship", "tournament",
                  "league", "season", "score", "goal", "medal",
                  "game", "games", "video", "gaming", "puzzle",
                  "toy", "adventure", "fantasy", "magic", "myth",
                  "legend", "fairy", "tale", "hero", "villain",
                  "character", "plot", "scene", "episode", "chapter",
                  "series", "edition", "collection", "gallery",
                  "museum", "exhibition", "award", "prize", "celebrity",
                  "star", "fan", "audience", "entertainment",
                  "fashion", "designer", "style", "magazine",
                  "newspaper", "journal", "blog", "broadcast",
                  "channel", "studio", "production", "director",
                  "producer", "actor", "actress", "singer",
                  "musician", "composer", "poet", "novelist",
                  "journalist", "photographer", "animator"},
    },
    # --- NEW CATEGORIES: capture the 1399-word misc dump ---
    {
        "id": "food_drink",
        "label": "a food or drink word — meal, breakfast, dinner, cook, fruit, bread, coffee, meat, rice, drink, taste",
        "seeds": {"meal", "meals", "breakfast", "lunch", "dinner", "snack",
                  "cook", "cooked", "cooking", "recipe", "recipes", "ingredient",
                  "diet", "appetite", "hungry", "taste", "flavor", "dish",
                  "fruit", "vegetable", "bread", "meat", "beef", "chicken",
                  "pork", "fish", "rice", "pasta", "soup", "salad", "sandwich",
                  "pizza", "burger", "cheese", "butter", "cream", "sauce",
                  "sugar", "salt", "pepper", "spice", "oil", "flour",
                  "egg", "eggs", "milk", "juice", "coffee", "tea", "wine",
                  "beer", "alcohol", "drink", "drinks", "water", "soda",
                  "cake", "pie", "chocolate", "candy", "cookie", "ice",
                  "honey", "nut", "nuts", "seed", "seeds", "grain"},
    },
    {
        "id": "health",
        "label": "a health or medical word — disease, patient, hospital, treatment, surgery, medicine, symptom, injury, cure, recovery",
        "seeds": {"disease", "diseases", "illness", "sick", "sickness",
                  "patient", "patients", "hospital", "hospitals", "clinic",
                  "treatment", "treatments", "therapy", "therapist",
                  "surgery", "surgeon", "operation", "procedure",
                  "medicine", "medication", "drug", "drugs", "dose",
                  "prescription", "pill", "vaccine", "antibiotic",
                  "symptom", "symptoms", "diagnosis", "condition",
                  "infection", "virus", "bacteria", "fever", "pain",
                  "injury", "injuries", "wound", "wounds", "bleeding",
                  "cure", "heal", "healing", "recovery", "recover",
                  "healthcare", "dental", "mental", "clinical",
                  "pregnant", "pregnancy", "birth", "nursing",
                  "cancer", "tumor", "stroke", "diabetes", "asthma",
                  "depression", "anxiety", "disorder", "disability",
                  "diet", "fitness", "exercise", "wellness", "healthy",
                  "blood", "bone", "tissue", "organ", "cell", "cells"},
    },
    {
        "id": "education",
        "label": "a learning or school word — school, student, teacher, class, lesson, degree, graduate, exam, study, university",
        "seeds": {"school", "schools", "university", "universities", "college",
                  "colleges", "campus", "academy", "institute", "institution",
                  "student", "students", "teacher", "teachers", "professor",
                  "tutor", "instructor", "principal", "dean",
                  "class", "classes", "classroom", "course", "courses",
                  "lesson", "lessons", "lecture", "lectures", "seminar",
                  "degree", "degrees", "diploma", "certificate",
                  "graduate", "graduated", "graduation", "undergraduate",
                  "exam", "exams", "test", "tests", "quiz", "assignment",
                  "homework", "essay", "thesis", "dissertation",
                  "study", "studied", "studying", "curriculum",
                  "scholarship", "academic", "education", "educational",
                  "semester", "grade", "grades", "gpa", "literacy",
                  "library", "textbook", "lab", "laboratory"},
    },
    {
        "id": "legal",
        "label": "a law or justice word — court, crime, judge, prison, trial, attorney, evidence, guilty, arrest, sentence",
        "seeds": {"court", "courts", "judge", "judges", "jury",
                  "trial", "trials", "verdict", "ruling", "appeal",
                  "attorney", "lawyer", "lawyers", "prosecutor",
                  "crime", "crimes", "criminal", "criminals",
                  "prison", "jail", "prisoner", "prisoners", "inmate",
                  "arrest", "arrested", "conviction", "convicted",
                  "guilty", "innocent", "suspect", "accused",
                  "sentence", "sentenced", "penalty", "fine", "bail",
                  "evidence", "witness", "testimony", "confession",
                  "murder", "theft", "robbery", "assault", "fraud",
                  "rape", "abuse", "harassment", "vandalism",
                  "lawsuit", "sue", "sued", "prosecution", "defense",
                  "parole", "probation", "warrant", "investigation",
                  "detective", "cop", "cops", "police", "officer",
                  "violation", "offense", "felony", "misdemeanor",
                  "legal", "illegal", "law", "laws", "legislation",
                  "regulation", "regulations", "compliance", "enforcement"},
    },
    {
        "id": "transport",
        "label": "a travel or vehicle word — car, bus, train, plane, drive, road, flight, trip, ticket, speed, travel",
        "seeds": {"car", "cars", "bus", "train", "trains", "plane", "planes",
                  "truck", "trucks", "vehicle", "vehicles", "van", "taxi", "cab",
                  "bike", "bicycle", "motorcycle", "boat", "ship", "ships",
                  "aircraft", "helicopter", "jet", "rocket",
                  "drive", "driver", "drivers", "driving", "rode", "ride",
                  "road", "roads", "highway", "freeway", "lane", "lanes",
                  "street", "streets", "route", "routes", "path",
                  "flight", "flights", "airport", "airline",
                  "trip", "trips", "journey", "travel", "traveled",
                  "destination", "departure", "arrival",
                  "ticket", "tickets", "fare", "pass", "passport",
                  "speed", "traffic", "parking", "garage",
                  "fuel", "gas", "gasoline", "engine", "engines",
                  "wheel", "wheels", "tire", "brake",
                  "station", "terminal", "port", "harbor",
                  "passenger", "passengers", "commute", "transit",
                  "crash", "accident", "collision", "wreck",
                  "mile", "miles", "kilometer", "distance"},
    },
    {
        "id": "money",
        "label": "a money or finance word — dollar, price, cost, payment, tax, profit, debt, budget, bank, investment, income",
        "seeds": {"dollar", "dollars", "cent", "cents", "penny",
                  "money", "cash", "coin", "coins", "currency",
                  "price", "prices", "cost", "costs", "fee", "fees",
                  "payment", "payments", "pay", "paid", "paying",
                  "tax", "taxes", "taxation",
                  "profit", "profits", "loss", "losses", "revenue",
                  "debt", "debts", "loan", "loans", "mortgage",
                  "budget", "budgets", "spending", "expense", "expenses",
                  "bank", "banks", "banking", "account", "accounts",
                  "investment", "investments", "investor", "investors",
                  "income", "salary", "wage", "wages", "earnings",
                  "stock", "stocks", "bond", "bonds", "share", "shares",
                  "credit", "credits", "debit", "interest",
                  "insurance", "pension", "retirement",
                  "wealth", "poverty", "rich", "poor",
                  "finance", "financial", "economy", "economic",
                  "inflation", "recession", "bankruptcy",
                  "trade", "trading", "commerce", "transaction",
                  "fund", "funds", "funding", "grant", "grants",
                  "asset", "assets", "equity", "capital"},
    },
    {
        "id": "government",
        "label": "a government or politics word — president, election, congress, vote, policy, reform, senator, nation, official, law",
        "seeds": {"president", "presidential", "governor", "mayor",
                  "senator", "senators", "congressman", "representative",
                  "politician", "politicians", "official", "officials",
                  "election", "elections", "vote", "votes", "voter", "voters",
                  "voting", "ballot", "campaign", "campaigns", "candidate",
                  "candidates", "nomination", "poll", "polls",
                  "congress", "senate", "parliament", "legislature",
                  "government", "governments", "federal", "state",
                  "policy", "policies", "reform", "reforms",
                  "legislation", "amendment", "constitution",
                  "democrat", "democrats", "republican", "republicans",
                  "liberal", "conservative", "socialist",
                  "administration", "cabinet", "ministry",
                  "district", "districts", "county", "counties",
                  "nation", "nations", "national", "international",
                  "sovereignty", "independence", "democracy",
                  "diplomacy", "diplomat", "embassy", "ambassador",
                  "treaty", "resolution", "sanction", "sanctions",
                  "immigration", "refugee", "refugees", "citizenship",
                  "authority", "authorities", "bureau", "agency",
                  "committee", "commission", "council", "board",
                  "political", "politics", "partisan", "bipartisan"},
    },
    {
        "id": "military",
        "label": "a conflict or military word — war, army, soldier, battle, attack, weapon, enemy, defense, troops, combat",
        "seeds": {"war", "wars", "warfare", "battle", "battles",
                  "army", "armies", "navy", "marine", "marines",
                  "soldier", "soldiers", "troop", "troops", "warrior",
                  "officer", "officers", "general", "colonel", "captain",
                  "commander", "lieutenant", "sergeant",
                  "attack", "attacks", "assault", "invasion", "raid",
                  "weapon", "weapons", "gun", "guns", "bomb", "bombs",
                  "missile", "missiles", "bullet", "bullets", "tank",
                  "rifle", "pistol", "grenade", "artillery",
                  "enemy", "enemies", "ally", "allies", "alliance",
                  "defense", "defend", "defended", "protect", "protection",
                  "combat", "fight", "fought", "fighting",
                  "victory", "defeat", "surrender", "retreat",
                  "mission", "operations", "strategy", "tactics",
                  "intelligence", "spy", "spies", "surveillance",
                  "military", "armed", "force", "forces",
                  "nuclear", "chemical", "explosive",
                  "conflict", "conflicts", "violence", "violent",
                  "terrorism", "terrorist", "terrorists", "hostage",
                  "casualty", "casualties", "veteran", "veterans",
                  "peace", "ceasefire", "truce", "negotiation"},
    },
    {
        "id": "digital",
        "label": "a technology or internet word — computer, software, app, website, data, online, code, email, screen, network",
        "seeds": {"computer", "computers", "laptop", "desktop",
                  "software", "hardware", "program", "programs",
                  "app", "apps", "application", "applications",
                  "website", "websites", "web", "page", "pages",
                  "data", "database", "file", "files", "folder",
                  "online", "offline", "internet", "wifi",
                  "code", "coding", "programming", "developer",
                  "email", "emails", "text", "message", "messages",
                  "screen", "screens", "display", "monitor",
                  "network", "networks", "server", "servers",
                  "cloud", "storage", "download", "upload",
                  "platform", "platforms", "system", "systems",
                  "device", "devices", "phone", "phones", "tablet",
                  "camera", "cameras", "video", "audio",
                  "digital", "virtual", "cyber", "tech", "technology",
                  "algorithm", "ai", "artificial", "robot", "automation",
                  "browser", "search", "click", "link", "links",
                  "blog", "post", "stream", "streaming",
                  "social", "media", "content", "user", "users",
                  "password", "security", "hack", "hacker",
                  "update", "version", "bug", "feature"},
    },
    {
        "id": "nature",
        "label": "a nature or environment word — tree, forest, mountain, river, ocean, rain, sun, animal, flower, weather, sky",
        "seeds": {"tree", "trees", "forest", "forests", "wood", "woods",
                  "jungle", "wilderness", "bush", "branch", "branches",
                  "leaf", "leaves", "root", "roots", "bark", "seed",
                  "flower", "flowers", "plant", "plants", "grass", "garden",
                  "mountain", "mountains", "hill", "hills", "valley",
                  "river", "rivers", "lake", "lakes", "pond", "stream",
                  "ocean", "sea", "beach", "shore", "coast", "island",
                  "desert", "cave", "cliff", "rock", "rocks", "soil",
                  "rain", "snow", "ice", "frost", "storm", "thunder",
                  "lightning", "flood", "drought", "earthquake",
                  "sun", "sunshine", "moon", "star", "stars", "sky",
                  "cloud", "clouds", "wind", "breeze", "fog",
                  "weather", "climate", "season", "spring", "autumn",
                  "animal", "animals", "bird", "birds", "fish",
                  "dog", "dogs", "cat", "cats", "horse", "horses",
                  "bear", "wolf", "lion", "tiger", "snake", "whale",
                  "insect", "butterfly", "bee", "spider",
                  "nature", "natural", "wild", "wildlife",
                  "environment", "environmental", "ecology", "ecosystem",
                  "pollution", "conservation", "endangered", "species"},
    },
    {
        "id": "body",
        "label": "a body part word — head, hand, face, eye, heart, arm, leg, foot, brain, skin, mouth, finger",
        "seeds": {"head", "heads", "face", "faces", "forehead",
                  "eye", "eyes", "ear", "ears", "nose", "mouth",
                  "lip", "lips", "tooth", "teeth", "tongue", "jaw",
                  "chin", "cheek", "throat", "neck",
                  "hand", "hands", "finger", "fingers", "thumb",
                  "palm", "fist", "wrist", "elbow",
                  "arm", "arms", "shoulder", "shoulders",
                  "leg", "legs", "knee", "knees", "ankle",
                  "foot", "feet", "toe", "toes", "heel",
                  "chest", "stomach", "belly", "waist", "hip",
                  "back", "spine", "rib", "ribs",
                  "heart", "lung", "lungs", "liver", "kidney",
                  "brain", "brains", "nerve", "nerves",
                  "skin", "bone", "bones", "muscle", "muscles",
                  "blood", "vein", "artery",
                  "hair", "beard", "eyebrow",
                  "body", "bodies", "flesh", "organ", "organs"},
    },
    {
        "id": "home",
        "label": "a home or living space word — house, apartment, room, kitchen, bedroom, furniture, door, window, rent, floor",
        "seeds": {"house", "houses", "home", "homes", "apartment",
                  "apartments", "flat", "condo", "mansion",
                  "room", "rooms", "bedroom", "bathroom", "kitchen",
                  "living", "dining", "hallway", "basement", "attic",
                  "garage", "porch", "balcony", "yard", "backyard",
                  "door", "doors", "window", "windows", "gate",
                  "wall", "walls", "floor", "floors", "ceiling",
                  "roof", "stair", "stairs", "step", "steps",
                  "furniture", "table", "chair", "chairs", "couch",
                  "sofa", "bed", "beds", "desk", "shelf", "shelves",
                  "cabinet", "drawer", "closet", "wardrobe",
                  "carpet", "curtain", "lamp", "mirror",
                  "shower", "bath", "sink", "toilet",
                  "rent", "rental", "tenant", "landlord",
                  "household", "domestic", "residential",
                  "property", "estate", "mortgage",
                  "neighborhood", "neighbor", "neighbors",
                  "address", "building", "buildings"},
    },
    {
        "id": "work",
        "label": "a work or career word — job, office, company, employee, manager, boss, project, meeting, hire, professional",
        "seeds": {"job", "jobs", "career", "careers", "profession",
                  "work", "works", "working", "worker", "workers",
                  "office", "offices", "workplace", "factory",
                  "company", "companies", "corporation", "firm", "firms",
                  "business", "businesses", "industry", "industries",
                  "employee", "employees", "employer", "employers",
                  "manager", "managers", "boss", "supervisor",
                  "staff", "team", "teams", "crew", "colleague",
                  "project", "projects", "task", "tasks", "assignment",
                  "meeting", "meetings", "conference",
                  "hire", "hired", "hiring", "fire", "fired",
                  "interview", "resume", "application",
                  "promotion", "raise", "bonus", "benefit",
                  "salary", "wage", "income", "paycheck",
                  "professional", "expertise", "experience",
                  "deadline", "schedule", "shift",
                  "retire", "retired", "retirement",
                  "unemployment", "layoff", "strike",
                  "productivity", "efficiency", "performance"},
    },
    {
        "id": "movement",
        "label": "a movement word — go, come, walk, run, move, drive, fly, sit, stand, turn, enter, leave, arrive",
        "seeds": {"go", "goes", "going", "went", "gone",
                  "come", "comes", "came", "coming",
                  "walk", "walked", "walking", "run", "ran", "running",
                  "move", "moved", "moves", "moving",
                  "drive", "drove", "driven", "driving",
                  "fly", "flew", "flying", "flown",
                  "ride", "rode", "riding",
                  "swim", "swam", "swimming",
                  "jump", "jumped", "jumping",
                  "climb", "climbed", "climbing",
                  "dance", "danced", "dancing",
                  "sit", "sat", "sitting", "stand", "stood", "standing",
                  "turn", "turned", "turning",
                  "enter", "entered", "entering",
                  "leave", "left", "leaving",
                  "return", "returned", "returning",
                  "arrive", "arrived", "arriving",
                  "depart", "departed",
                  "fall", "fell", "falling", "fallen",
                  "rise", "rose", "rising", "risen",
                  "slide", "slip", "crawl", "rush",
                  "escape", "flee", "chase", "follow",
                  "wander", "roam", "travel", "explore",
                  "approach", "retreat", "advance",
                  "pass", "passed", "passing",
                  "cross", "crossed", "crossing",
                  "step", "stepped", "stepping",
                  "reach", "reached", "reaching",
                  "land", "landed", "landing",
                  "drop", "dropped", "sink", "sank"},
    },
    {
        "id": "creation",
        "label": "a making or building word — make, build, create, design, produce, develop, form, write, invent, generate",
        "seeds": {"make", "makes", "made", "making",
                  "build", "built", "building",
                  "create", "created", "creating", "creation",
                  "design", "designed", "designing",
                  "produce", "produced", "producing", "production",
                  "develop", "developed", "developing", "development",
                  "form", "formed", "forming",
                  "construct", "constructed", "construction",
                  "manufacture", "manufactured", "manufacturing",
                  "write", "wrote", "written", "writing",
                  "draw", "drew", "drawn", "drawing",
                  "paint", "painted", "painting",
                  "compose", "composed",
                  "invent", "invented", "invention",
                  "generate", "generated",
                  "establish", "established",
                  "found", "founded", "foundation",
                  "launch", "launched",
                  "assemble", "assembled",
                  "craft", "crafted",
                  "shape", "shaped", "shaping",
                  "organize", "organized",
                  "prepare", "prepared", "preparing",
                  "install", "installed",
                  "publish", "published",
                  "record", "recorded", "recording"},
    },
    {
        "id": "change",
        "label": "a change or transformation word — change, grow, improve, increase, reduce, fix, update, adjust, become, develop",
        "seeds": {"change", "changed", "changes", "changing",
                  "grow", "grew", "grown", "growing", "growth",
                  "improve", "improved", "improving", "improvement",
                  "increase", "increased", "increasing",
                  "decrease", "decreased", "decreasing",
                  "reduce", "reduced", "reducing", "reduction",
                  "expand", "expanded", "expanding", "expansion",
                  "shrink", "shrunk",
                  "fix", "fixed", "fixing",
                  "repair", "repaired",
                  "update", "updated", "updating",
                  "adjust", "adjusted", "adjusting",
                  "modify", "modified",
                  "transform", "transformed", "transformation",
                  "convert", "converted", "conversion",
                  "adapt", "adapted", "adapting",
                  "evolve", "evolved", "evolution",
                  "shift", "shifted", "shifting",
                  "replace", "replaced", "replacing",
                  "restore", "restored",
                  "recover", "recovered",
                  "reform", "reformed",
                  "revise", "revised",
                  "edit", "edited", "editing",
                  "become", "became", "becoming",
                  "turn", "turned",
                  "rise", "rose", "rising",
                  "decline", "declined", "declining",
                  "progress", "progressed",
                  "advance", "advanced", "advancing"},
    },
    {
        "id": "names",
        "label": "a name — a person's name, city, country, or place name like John, Paris, London, America",
        "seeds": {
            "aaron", "alan", "alice", "anderson", "andrew", "anne", "anthony",
            "arthur", "austin", "baker", "barry", "bell", "bobby", "brian",
            "cameron", "campbell", "carter", "charlie", "christopher", "cooper",
            "daniel", "danny", "davis", "duke", "evans", "francis", "gordon",
            "greg", "harris", "harry", "howard", "jacob", "jason", "jerry",
            "jim", "jimmy", "johnny", "jon", "jonathan", "jordan", "joseph",
            "josh", "justin", "kennedy", "kevin", "kim", "laura", "lawrence",
            "lee", "leo", "lincoln", "margaret", "maria", "mario", "marshall",
            "martin", "matt", "matthew", "mitchell", "moore", "murray",
            "oliver", "oscar", "parker", "pat", "patrick", "phil", "philip",
            "ross", "ryan", "santa", "scott", "sean", "steve", "stewart",
            "thompson", "walker", "wilson",
            "angeles", "asia", "barcelona", "berlin", "britain", "brooklyn",
            "cambridge", "chelsea", "chicago", "colorado", "dc", "delhi",
            "denver", "edinburgh", "georgia", "houston", "jersey", "kansas",
            "las", "liverpool", "manchester", "massachusetts", "minnesota",
            "missouri", "moscow", "netherlands", "nigeria", "oklahoma",
            "oxford", "philadelphia", "philippines", "rio", "scotland",
            "sydney", "sweden", "syria", "thailand", "turkey", "wisconsin",
            "zealand", "gop", "christ", "christians",
        },
    },
    # --- MISC: everything else ---
    {
        "id": "misc",
        "label": "another word not in the above categories",
        "seeds": set(),
    },
]


# WordNet routing for words without a seed (2026-09-21). Order matters: a word is placed by the
# first rule that fires. Domain rules look at the most common sense's hypernym closure; the rest use
# WordNet's lexicographer file of that sense. Off by default: the routed tree cost +46% calls at n=4
# (PAPER F38); the flat semantic split below cost +27% (F40). VOCAB_ROUTE=1 / VOCAB_FLAT=1 enable them.
from nltk.corpus import wordnet as wn  # noqa: E402

VOCAB_ROUTE = os.environ.get("VOCAB_ROUTE", "0") == "1"
VOCAB_FLAT = os.environ.get("VOCAB_FLAT", "0") == "1"

DOMAIN_RULES = [
    ("health", {"health_professional", "medical_practitioner", "disease", "illness", "medicine", "drug",
                "hospital", "symptom", "medical_procedure", "injury", "ill_health", "therapy", "pathology"}),
    ("legal", {"law", "legal_document", "court", "crime", "criminal", "offense", "lawyer", "judge",
               "legal_action", "jurisprudence", "police", "punishment"}),
    ("military", {"military", "military_service", "weapon", "war", "armed_forces", "soldier", "military_action",
                  "combat", "arm", "battle", "military_unit", "weaponry"}),
    ("government", {"government", "politics", "political_leader", "legislature", "election", "policy",
                    "political_party", "politician", "head_of_state", "legislator", "diplomacy"}),
    ("money", {"monetary_unit", "payment", "money", "finance", "financial_institution", "cost", "income",
               "tax", "debt", "currency", "price", "financial_gain", "loss", "commerce", "trade"}),
    ("transport", {"vehicle", "conveyance", "road", "travel", "journey", "aircraft", "vessel", "ship",
                   "public_transport", "driver", "airport", "traffic"}),
    ("digital", {"computer", "software", "computer_network", "website", "internet", "computer_program",
                 "electronic_device", "computer_science", "data_processor", "code", "hardware", "digital_computer"}),
    ("science", {"science", "scientific_discipline", "chemistry", "physics", "biology", "chemical_element",
                 "scientist", "experiment", "natural_science", "mathematics", "chemical", "molecule", "atom",
                 "energy", "radiation", "gene", "cell"}),
    ("food_drink", {"food", "nutrient", "beverage", "dish", "meal", "foodstuff", "drink", "fruit", "vegetable",
                    "edible_fruit", "cooking", "cook", "bread", "meat", "dairy_product", "alcohol"}),
    ("education", {"educational_institution", "education", "student", "teacher", "school", "course",
                   "lesson", "learning", "degree", "examination", "educator"}),
    ("arts", {"art", "music", "musical_instrument", "film", "literature", "creative_person", "artist",
              "musician", "performance", "entertainment", "sport", "athlete", "game", "song", "genre",
              "show", "dance", "drawing", "painting", "novel", "poem", "writer"}),
    ("home", {"room", "furniture", "dwelling", "housing", "house", "household", "appliance", "kitchen",
              "bed", "garden", "building"}),
    ("body", {"body_part", "organ", "body_substance", "body_covering", "anatomical_structure", "tissue",
              "external_body_part", "bone"}),
    ("nature", {"animal", "plant", "natural_object", "geological_formation", "weather", "body_of_water",
                "atmospheric_phenomenon", "tree", "flower", "bird", "fish", "insect", "mammal", "land",
                "natural_phenomenon", "sky", "season", "star", "planet", "sun"}),
    ("work", {"occupation", "job", "profession", "employee", "employer", "worker", "workplace", "business",
              "enterprise", "manager", "labor", "career", "office", "skilled_worker"}),
    ("world", {"location", "region", "country", "city", "town", "district", "geographical_area", "continent",
               "state", "land", "area", "territory", "place"}),
]
LEXNAME_RULES = {
    "noun.person": "people", "noun.group": "society", "noun.feeling": "feeling", "verb.emotion": "feeling",
    "noun.cognition": "think", "verb.cognition": "think", "verb.perception": "think",
    "noun.communication": "communicate", "verb.communication": "communicate",
    "verb.motion": "movement", "verb.creation": "creation", "verb.change": "change",
    "noun.time": "event_time", "noun.event": "event_time",
    "noun.location": "world", "noun.artifact": "object", "noun.object": "object", "noun.substance": "object",
    "noun.food": "food_drink", "noun.body": "body", "noun.animal": "nature", "noun.plant": "nature",
    "noun.phenomenon": "nature", "noun.possession": "money", "noun.quantity": "number",
    "noun.attribute": "concept", "noun.state": "concept", "noun.relation": "concept", "noun.process": "concept",
    "noun.motive": "concept", "noun.shape": "concept", "noun.act": "action",
    "adj.all": "describe", "adj.pert": "describe", "adj.ppl": "describe", "adv.all": "time_manner",
}


def wordnet_category(w):
    ss = wn.synsets(w)
    if not ss:
        return None
    s = ss[0]  # the most common sense
    if s.pos() == "n":
        names = {h.name().split(".")[0] for h in s.closure(lambda x: x.hypernyms())} | {s.name().split(".")[0]}
        for cid, keys in DOMAIN_RULES:
            if names & keys:
                return cid
    if s.pos() == "v":
        return LEXNAME_RULES.get(s.lexname(), "action")
    return LEXNAME_RULES.get(s.lexname())


def assign_words():
    assignments = {}
    for cat in CATEGORIES:
        for w in cat["seeds"]:
            if w in WORD_SET and w not in assignments:
                assignments[w] = cat["id"]

    for w in ALL_WORDS:
        if w in assignments:
            continue
        pos = WORD_POS.get(w, "NN")
        assigned = False
        for cat in CATEGORIES:
            if "pos_match" in cat and pos in cat["pos_match"]:
                assignments[w] = cat["id"]
                assigned = True
                break
        if not assigned and VOCAB_ROUTE:
            cid = wordnet_category(w)
            if cid:
                assignments[w] = cid
                assigned = True
        if not assigned:
            if pos.startswith("VB"):
                assignments[w] = "action"
            elif pos.startswith("JJ"):
                assignments[w] = "describe"
            elif pos.startswith("RB"):
                assignments[w] = "time_manner"
            else:
                routed = False
                for suf, cat in [
                    ("tion", "concept"), ("sion", "concept"),
                    ("ment", "concept"), ("ness", "concept"),
                    ("ity", "concept"), ("ism", "concept"),
                    ("ence", "concept"), ("ance", "concept"),
                ]:
                    if w.endswith(suf) and len(w) > len(suf) + 2:
                        assignments[w] = cat
                        routed = True
                        break
                if not routed:
                    assignments[w] = "misc"
    return assignments


# Semantic split (2026-09-21, PAPER F38): a category over MAX_CHILDREN becomes several flat
# top-level groups that share a WordNet meaning, so every content word is one navigate call and the
# classifier chooses among meanings, never among letter ranges. Groups too small to stand alone
# merge into "other"; what is still too big is cut into frequency bands whose labels list their words.
MIN_GROUP = 25
MAX_GROUP = 250  # the API allows 255 options per question; the say question stays near 200
LEX_NAMES = {  # WordNet keys as the classifier should read them
    "noun.cognition": "ideas and knowledge", "verb.cognition": "thinking and knowing",
    "verb.perception": "seeing and hearing", "verb.communication": "saying and telling",
    "noun.communication": "things said or written", "noun.act": "things done",
    "verb.social": "doing things with others", "verb.possession": "having, getting and giving",
    "verb.stative": "being and staying", "verb.motion": "moving", "verb.change": "changing",
    "verb.contact": "touching and handling", "verb.competition": "competing", "verb.body": "the body doing",
    "verb.consumption": "eating and using", "verb.creation": "making", "verb.emotion": "feeling",
    "noun.person": "people", "noun.group": "groups", "relative": "family", "leader": "leaders",
    "worker": "workers", "adult": "men and women", "communicator": "speakers and writers",
    "noun.location": "places", "noun.artifact": "made things", "noun.object": "natural things",
    "noun.substance": "materials", "noun.food": "food", "noun.body": "the body", "noun.attribute": "qualities",
    "noun.state": "states and conditions", "noun.relation": "relations", "noun.event": "events",
    "noun.time": "times", "noun.quantity": "amounts", "noun.possession": "belongings", "noun.process": "processes",
    "size": "size", "other": "other", "unknown": "other",
}
_SENSES = {}


def _sense(w):
    if w not in _SENSES:
        ss = wn.synsets(w)
        _SENSES[w] = ss[0] if ss else None
    return _SENSES[w]


def _keys(w):
    """Meaning keys from coarse to fine: lexicographer file, then the hypernym path from the root.
    Adjectives: their attribute (size, age, quality ...)."""
    s = _sense(w)
    if s is None:
        return ["unknown"]
    if s.pos() in ("a", "s"):
        attrs = s.attributes() or (s.similar_tos()[0].attributes() if s.similar_tos() else [])
        return ["adjective", attrs[0].name().split(".")[0] if attrs else "other"]
    if s.pos() == "r":
        return ["adverb"]
    paths = s.hypernym_paths()
    path = [x.name().split(".")[0] for x in max(paths, key=len)] if paths else []
    return [s.lexname()] + path[2:]  # skip entity / physical_entity-abstraction


def _partition(words, depth, max_c):
    """Groups of <= max_c words, split by the key at `depth`, deeper where still too big."""
    if len(words) <= max_c:
        return [(None, words)]
    groups = {}
    for w in words:
        ks = _keys(w)
        groups.setdefault(ks[depth] if depth < len(ks) else "other", []).append(w)
    if len(groups) == 1 and depth < 10:
        return _partition(words, depth + 1, max_c)
    out = []
    for key, ws in groups.items():
        for sub_key, sub in _partition(ws, depth + 1, max_c):
            out.append((sub_key or key, sub))
    return out


def semantic_groups(words, max_c=MAX_GROUP):
    """Named groups of <= max_c words for a category too big for one node. Returns
    [(name, words)] in frequency order of their first word; small groups merge into 'other'."""
    rank = {w: i for i, w in enumerate(ALL_WORDS)}
    big, small = [], []
    for key, ws in _partition(words, 0, max_c):
        (big if len(ws) >= MIN_GROUP else small).append((key, ws))
    other = [w for _, ws in small for w in ws]
    for i in range(0, len(other), max_c):
        big.append(("other", other[i:i + max_c]))
    merged = {}
    for key, ws in big:  # same key from different branches, or several bands, join when they fit
        for k in (key, f"{key} more"):
            if len(merged.get(k, [])) + len(ws) <= max_c:
                merged.setdefault(k, []).extend(ws)
                break
        else:
            merged[f"{key} {len(merged)}"] = list(ws)
    groups = [(k, sorted(ws, key=rank.get)) for k, ws in merged.items()]
    return sorted(groups, key=lambda g: rank.get(g[1][0], 10**9))


def top_level_entries(cat, words, max_c=MAX_GROUP):
    """One flat entry, or several for a category over max_c: (id, label, words) triples."""
    if len(words) <= max_c or not VOCAB_FLAT:
        return [(cat["id"], cat["label"], words)]
    head = cat["label"].split(" — ")[0]
    entries = []
    for key, ws in semantic_groups(words, max_c):
        base, _, more = key.partition(" ")
        name = LEX_NAMES.get(base, base.replace("_", " ")) + (" (more)" if more else "")
        cid = f"{cat['id']}_{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}"
        entries.append((cid, f"{head}, {name} — {', '.join(ws[:8])}", ws))
    return entries


def recursive_split(words, max_c=MAX_CHILDREN, char_pos=0):
    """Recursively split words into groups of <= max_c by character position."""
    if len(words) <= max_c:
        return [{"type": "leaf", "word": w} for w in sorted(words)]

    # Try splitting by character at char_pos
    by_char = {}
    for w in words:
        ch = w[char_pos] if char_pos < len(w) else "_"
        by_char.setdefault(ch, []).append(w)

    # If splitting didn't help (all same char), try next position
    if len(by_char) <= 1:
        if char_pos < 10:
            return recursive_split(words, max_c, char_pos + 1)
        # Fall back to mechanical chunking
        sw = sorted(words)
        nodes = []
        for i in range(0, len(sw), max_c):
            chunk = sw[i:i+max_c]
            nodes.append({
                "type": "branch",
                "id": f"{chunk[0]}..{chunk[-1]}",
                "label": f"{', '.join(chunk[:5])}",
                "children": [{"type": "leaf", "word": w} for w in chunk],
            })
        return nodes

    # Merge adjacent small char groups
    sorted_keys = sorted(by_char.keys())
    groups = []
    cur_keys = []
    cur_words = []
    for k in sorted_keys:
        kw = by_char[k]
        if cur_words and len(cur_words) + len(kw) > max_c:
            groups.append((list(cur_keys), list(cur_words)))
            cur_keys = []
            cur_words = []
        cur_keys.append(k)
        cur_words.extend(kw)
    if cur_words:
        groups.append((list(cur_keys), list(cur_words)))

    nodes = []
    for keys, grp in groups:
        tag = keys[0] if len(keys) == 1 else f"{keys[0]}-{keys[-1]}"
        sample = ", ".join(sorted(grp)[:5])
        if len(grp) <= max_c:
            nodes.append({
                "type": "branch",
                "id": tag,
                "label": f"{tag}: {sample}",
                "children": [{"type": "leaf", "word": w} for w in sorted(grp)],
            })
        else:
            sub = recursive_split(grp, max_c, char_pos + 1)
            nodes.append({
                "type": "branch",
                "id": tag,
                "label": f"{tag}: {sample}",
                "children": sub,
            })

    # If we still have too many groups, chunk them
    if len(nodes) > max_c:
        chunks = []
        for i in range(0, len(nodes), max_c):
            chunk = nodes[i:i+max_c]
            if len(chunk) == 1:
                chunks.append(chunk[0])
            else:
                fid = chunk[0]["id"]
                lid = chunk[-1]["id"]
                chunks.append({
                    "type": "branch",
                    "id": f"{fid}..{lid}",
                    "label": f"{fid} to {lid}",
                    "children": chunk,
                })
        return chunks

    return nodes


def build_and_save():
    assignments = assign_words()

    cat_words = {}
    for w, cid in assignments.items():
        cat_words.setdefault(cid, []).append(w)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Clean old files
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    index = {"categories": []}

    for cat in CATEGORIES:
        words = sorted(cat_words.get(cat["id"], []))
        if not words:
            continue

        for cid, label, ws in top_level_entries(cat, words):
            cat_data = {
                "id": cid,
                "label": label,
                "word_count": len(ws),
                "children": recursive_split(ws, MAX_GROUP if VOCAB_FLAT else MAX_CHILDREN),
            }
            with open(os.path.join(OUTPUT_DIR, f"{cid}.json"), 'w') as f:
                json.dump(cat_data, f, indent=2)
            index["categories"].append({
                "id": cid,
                "label": label,
                "file": f"{cid}.json",
                "word_count": len(ws),
            })

    with open(os.path.join(OUTPUT_DIR, "_index.json"), 'w') as f:
        json.dump(index, f, indent=2)

    # Stats
    total = sum(len(cat_words.get(c["id"], [])) for c in CATEGORIES)
    print(f"Vocab tree v3: {OUTPUT_DIR}/")
    print(f"  Total: {total} / {len(ALL_WORDS)}")
    print(f"  Categories: {len(index['categories'])}")

    def max_ch(node):
        if node.get("type") == "leaf": return 0
        mc = len(node.get("children", []))
        for c in node.get("children", []):
            mc = max(mc, max_ch(c))
        return mc

    def leaf_depths(node, d=0):
        if node.get("type") == "leaf": return [d]
        ds = []
        for c in node.get("children", []): ds.extend(leaf_depths(c, d+1))
        return ds

    print()
    for entry in index["categories"]:
        with open(os.path.join(OUTPUT_DIR, entry["file"])) as f:
            data = json.load(f)
        mc = max_ch(data)
        ds = leaf_depths(data)
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, entry["file"]))
        print(f"  {entry['id']:15s}  {entry['word_count']:5d} words  max_ch={mc:2d}  depth={min(ds)}-{max(ds)}  {sz/1024:.1f}KB")


if __name__ == "__main__":
    build_and_save()
