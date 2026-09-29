#!/usr/bin/env python3
"""Per-language check: gemma vs qwen3 embedder, same 10 queries translated by hand.
Queries are translated, corpus stays as in model_eval.py (mostly English), so this
measures cross-lingual query->doc retrieval. Run from anywhere:
    python scripts/eval/multilingual_eval.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import model_eval as m

T = {
"fr": ["pourquoi ne puis-je pas simplement changer le modèle d'embedding lors d'une mise à jour de l'application",
 "combien d'espace de stockage un million de vecteurs de dimension 1024 occupent-ils",
 "à quoi sert la constante k dans la fusion par rang réciproque",
 "pourquoi la recherche par mots-clés seule rate les questions reformulées",
 "faire dire au système qu'il n'a rien trouvé au lieu de renvoyer des passages non pertinents",
 "puis-je remplacer le reranker sans reconstruire l'index",
 "combien de candidats envoyer au cross-encoder sur un portable sans GPU",
 "quel est le problème quand on découpe les documents tous les N tokens",
 "envoyer au LLM le passage plus large qui entoure, mais chercher sur les petits",
 "combien de questions de référence faut-il pour commencer à évaluer"],
"de": ["warum kann ich das Embedding-Modell nicht einfach bei einem App-Update wechseln",
 "wie viel Speicher brauchen eine Million Vektoren mit 1024 Dimensionen",
 "was bewirkt die Konstante k bei der Reciprocal Rank Fusion",
 "warum die reine Stichwortsuche umformulierte Fragen verpasst",
 "das System sagen lassen, dass es nichts gefunden hat, statt irrelevante Passagen zurückzugeben",
 "kann ich den Reranker ersetzen, ohne den Index neu aufzubauen",
 "wie viele Kandidaten soll ich auf einem Laptop ohne GPU an den Cross-Encoder schicken",
 "was ist das Problem, wenn man Dokumente alle N Tokens zerschneidet",
 "den größeren umgebenden Abschnitt an das LLM schicken, aber auf kleinen suchen",
 "wie viele Referenzfragen brauche ich, um mit der Evaluierung zu beginnen"],
"es": ["por qué no puedo simplemente cambiar el modelo de embeddings en una actualización de la app",
 "cuánto almacenamiento ocupan un millón de vectores de 1024 dimensiones",
 "para qué sirve la constante k en la fusión por rango recíproco",
 "por qué la búsqueda solo por palabras clave falla con preguntas parafraseadas",
 "hacer que el sistema diga que no encontró nada en lugar de devolver pasajes irrelevantes",
 "puedo reemplazar el reranker sin reconstruir el índice",
 "cuántos candidatos enviar al cross-encoder en un portátil sin GPU",
 "cuál es el problema de cortar los documentos cada N tokens",
 "enviar al LLM el pasaje más grande que rodea, pero buscar con los pequeños",
 "cuántas preguntas de referencia necesito para empezar a evaluar"],
"pt": ["por que não posso simplesmente trocar o modelo de embeddings numa atualização do app",
 "quanto armazenamento ocupam um milhão de vetores de 1024 dimensões",
 "para que serve a constante k na fusão por classificação recíproca",
 "por que a busca apenas por palavras-chave falha em perguntas reformuladas",
 "fazer o sistema dizer que não encontrou nada em vez de devolver trechos irrelevantes",
 "posso substituir o reranker sem reconstruir o índice",
 "quantos candidatos enviar ao cross-encoder num portátil sem GPU",
 "qual é o problema de cortar documentos a cada N tokens",
 "enviar ao LLM o trecho maior ao redor, mas pesquisar nos pequenos",
 "quantas perguntas de referência preciso para começar a avaliar"],
"ru": ["почему нельзя просто сменить модель эмбеддингов при обновлении приложения",
 "сколько места занимает миллион векторов размерности 1024",
 "за что отвечает константа k в слиянии по взаимным рангам",
 "почему поиск только по ключевым словам не находит перефразированные вопросы",
 "сделать так, чтобы система говорила, что ничего не нашла, а не возвращала нерелевантные фрагменты",
 "можно ли заменить реранкер без перестроения индекса",
 "сколько кандидатов отправлять кросс-энкодеру на ноутбуке без GPU",
 "в чём проблема разрезания документов каждые N токенов",
 "отправлять в LLM более крупный окружающий фрагмент, а искать по маленьким",
 "сколько эталонных вопросов нужно, чтобы начать оценку"],
"zh": ["为什么不能在应用更新时直接更换嵌入模型",
 "一百万个1024维向量需要多少存储空间",
 "倒数排名融合中的常数k有什么作用",
 "为什么仅靠关键词搜索会漏掉换了说法的问题",
 "让系统说没有找到，而不是返回不相关的段落",
 "可以在不重建索引的情况下更换重排序模型吗",
 "在没有GPU的笔记本上应该给交叉编码器发送多少候选",
 "每隔N个词元切分文档有什么问题",
 "把更大的上下文段落发给大模型，但用小段落做检索",
 "开始评估需要多少个标准问题"],
"ja": ["アプリの更新時に埋め込みモデルを単純に変更できないのはなぜですか",
 "1024次元のベクトル100万件はどれくらいの容量になりますか",
 "逆順位融合における定数kは何をしますか",
 "キーワード検索だけでは言い換えた質問を見逃すのはなぜですか",
 "無関係な文章を返す代わりに、見つからなかったとシステムに答えさせたい",
 "インデックスを再構築せずにリランカーを入れ替えられますか",
 "GPUのないノートパソコンでは、クロスエンコーダにいくつ候補を送るべきですか",
 "N個のトークンごとに文書を切ることの問題点は何ですか",
 "小さな断片で検索し、それを囲む大きな箇所をLLMに渡したい",
 "評価を始めるには正解付きの質問がいくつ必要ですか"],
"ar": ["لماذا لا يمكنني ببساطة تغيير نموذج التضمين عند تحديث التطبيق",
 "كم تبلغ مساحة التخزين اللازمة لمليون متجه بأبعاد 1024",
 "ما وظيفة الثابت k في دمج الترتيب التبادلي",
 "لماذا يفشل البحث بالكلمات المفتاحية وحده مع الأسئلة المعاد صياغتها",
 "جعل النظام يقول إنه لم يجد شيئًا بدل إرجاع مقاطع غير ذات صلة",
 "هل يمكنني استبدال معيد الترتيب دون إعادة بناء الفهرس",
 "كم مرشحًا يجب إرساله إلى المشفّر المتقاطع على حاسوب محمول بدون معالج رسومي",
 "ما المشكلة في تقطيع المستندات كل N رمز",
 "إرسال المقطع الأكبر المحيط إلى النموذج اللغوي مع البحث بالمقاطع الصغيرة",
 "كم سؤالًا مرجعيًا أحتاج لبدء التقييم"],
}
base = m.QUERIES[:10]
T["en"] = [q for q, _ in base]

corpus = m.build_corpus()
res = {}
for name in ("qwen3-emb-0.6B-q8", "embeddinggemma-300M-q4"):
    path, pool, qp, dp = m.EMBEDDERS[name]
    s = m.Server(path, 8770, ["--embedding", *(["--pooling", pool] if pool else [])])
    dv = m.embed(8770, [dp + c["text"] for c in corpus])
    for lang, qs in T.items():
        qv = m.embed(8770, [qp + q for q in qs])
        ranks = [sorted(range(len(corpus)), key=lambda i: -sum(a * b for a, b in zip(q, dv[i]))) for q in qv]
        m.QUERIES = base
        res[(name, lang)] = m.metrics(ranks, corpus)
    s.stop()
langs = list(T)
print("| lang | qwen3 hit@1 | gemma hit@1 | qwen3 MRR | gemma MRR |\n|---|---|---|---|---|")
for l in langs:
    q, g = res[("qwen3-emb-0.6B-q8", l)], res[("embeddinggemma-300M-q4", l)]
    print(f"| {l} | {q['hit@1']:.1f} | {g['hit@1']:.1f} | {q['MRR']:.2f} | {g['MRR']:.2f} |")
avg = lambda n, k: sum(res[(n, l)][k] for l in langs if l != "en") / (len(langs) - 1)
print(f"| non-EN avg | {avg('qwen3-emb-0.6B-q8','hit@1'):.2f} | {avg('embeddinggemma-300M-q4','hit@1'):.2f} | {avg('qwen3-emb-0.6B-q8','MRR'):.2f} | {avg('embeddinggemma-300M-q4','MRR'):.2f} |")
