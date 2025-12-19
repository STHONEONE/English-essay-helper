import nltk

import streamlit as st
import re
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.corpus import wordnet


# --- 0. 环境配置与数据下载 ---
@st.cache_resource
def download_nltk_data():
    resources = ['wordnet', 'omw-1.4', 'punkt', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


download_nltk_data()

# --- 1. 页面配置 ---
st.set_page_config(page_title="Essay Optimizer AI Final", layout="wide", page_icon="🎓")

st.title("🎓 AI 英语作文深度优化助手")
st.markdown("核心功能：**智能词汇列表**  + **长难句自动拆分**")

# --- 停用词 ---
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'up', 'about', 'into', 'over', 'after', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'it', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your',
    'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them', 'so', 'very', 'really'
}


# --- 核心函数 ---

def get_synonyms(word):
    """获取同义词"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            cleaned_syn = l.name().replace('_', ' ')
            if cleaned_syn.lower() != word.lower():
                synonyms.add(cleaned_syn)
    return list(synonyms)[:5]


def smart_split_sentence(sentence):
    """
    长难句拆分算法 (保留 v3.1 的逻辑)
    """
    split_pattern = r'(,\s*(?:but|and|so|because|although|since|while))'
    parts = re.split(split_pattern, sentence, flags=re.IGNORECASE)

    if len(parts) == 1: return None

    refined_sentences = []
    current_sent = parts[0]

    for i in range(1, len(parts), 2):
        separator = parts[i]
        next_part = parts[i + 1]
        conjunction = re.sub(r'[,\s]', '', separator)
        refined_sentences.append(current_sent.strip() + ".")
        current_sent = f"{conjunction.capitalize()} {next_part.strip()}"

    refined_sentences.append(current_sent.strip())
    return refined_sentences


# --- 侧边栏 ---
with st.sidebar:
    st.header("📝 作文输入")
    default_text = "The rain was so big that our clothes were all wet, and we couldn't find the bus stop because it was too dark, but finally we walked home tiredly. It was a good day and we had a good time."
    text_input = st.text_area("在此粘贴作文:", value=default_text, height=300)
    analyze_btn = st.button("✨ 启动深度优化")

# --- 主逻辑 ---
if analyze_btn and text_input:
    blob = TextBlob(text_input)

    # ---------------------------
    # 模块一：全局数据
    # ---------------------------
    st.subheader("📊 全局诊断")
    col1, col2, col3 = st.columns(3)

    words = re.findall(r'\b\w+\b', text_input.lower())
    filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    col1.metric("情感指数", f"{blob.sentiment.polarity:.2f}")
    col2.metric("总单词数", len(words))
    col3.metric("词汇丰富度", f"{len(set(words)) / len(words):.1%}" if words else "0%")

    st.divider()

    # ---------------------------
    # 模块二：词汇优化 (已改回 v3.0 列表 UI)
    # ---------------------------
    st.subheader("💡 深度优化建议 (Optimization Suggestions)")
    st.markdown("#### 1. 词汇升级 (Vocabulary Upgrade)")

    # 这一块完全还原了你截图中的设计：蓝色背景提示 + 列表展示
    st.info("检测你使用频率较高或过于简单的词，并推荐高级替换词：")

    word_counts = Counter(filtered_words)
    common_words = word_counts.most_common(5)

    if not common_words:
        st.write("👏 词汇使用非常多样，没有发现明显的重复用词！")
    else:
        # 生成列表数据
        for word, count in common_words:
            syns = get_synonyms(word)
            if syns:
                syns_str = ", ".join(syns)
                # 这种格式就是你想要的效果：
                st.markdown(f"- **「{word}」** (用了{count}次) 👉 可替换为: *{syns_str}*")
            else:
                st.markdown(f"- **「{word}」** (用了{count}次) - 暂无同义词推荐")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------
    # 模块三：长难句自动拆分 (保留 v3.1 拆分功能)
    # ---------------------------
    st.markdown("#### 2. 长难句智能拆分 (Smart Sentence Splitter)")

    LONG_SENTENCE_THRESHOLD = 20
    long_sentences = []

    for sentence in blob.sentences:
        if len(sentence.words) > LONG_SENTENCE_THRESHOLD:
            long_sentences.append(str(sentence))

    if not long_sentences:
        st.success("✅ 句子结构良好，没有发现过长句子。")
    else:
        st.warning(f"⚠️ 发现 {len(long_sentences)} 个长难句，AI 生成了拆分方案：")

        for i, raw_sent in enumerate(long_sentences, 1):
            with st.expander(f"🚩 长句 {i} (点击查看拆分结果)", expanded=True):
                st.markdown("**🔴 原句 (Original):**")
                st.info(raw_sent)

                split_result = smart_split_sentence(raw_sent)

                st.markdown("**🟢 AI 建议拆分 (Suggested Split):**")
                if split_result:
                    for part in split_result:
                        st.success(part)
                    st.caption("💡 算法原理：基于连词 (and, but, because) 识别逻辑断点并重组。")
                else:
                    st.error("❌ 这句话结构太复杂，AI 无法自动安全拆分，请人工修改。")

elif not text_input:

    st.info("👈 请在左侧输入作文开始分析")
