"""
====================================================
  SMS Spam Classifier — NLP & Machine Learning
  By: Ahmed Adel Shosha
  Dataset: Based on UCI SMS Spam Collection
  5,572 real SMS messages
====================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import string
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   SMS SPAM CLASSIFIER — NLP & Machine Learning")
print("   By: Ahmed Adel Shosha")
print("   Dataset: UCI SMS Spam Collection (5,572 messages)")
print("=" * 60)

# =====================================================
# 1. LOAD DATASET
# =====================================================
print("\n[1] Loading dataset...")

df = pd.read_csv('/home/claude/sms_spam.csv')
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"   Total messages : {len(df):,}")
print(f"   Ham  (legit)   : {df['label'].value_counts()['ham']:,} ({df['label'].value_counts()['ham']/len(df)*100:.1f}%)")
print(f"   Spam           : {df['label'].value_counts()['spam']:,} ({df['label'].value_counts()['spam']/len(df)*100:.1f}%)")

# =====================================================
# 2. EDA — EXPLORATORY DATA ANALYSIS
# =====================================================
print("\n[2] Exploratory Data Analysis...")

df['msg_length']    = df['message'].apply(len)
df['word_count']    = df['message'].apply(lambda x: len(x.split()))
df['has_number']    = df['message'].apply(lambda x: any(c.isdigit() for c in x)).astype(int)
df['has_currency']  = df['message'].apply(lambda x: any(c in x for c in ['£','$','€','cash','prize','win'])).astype(int)
df['has_uppercase'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper())).astype(int)
df['exclamations']  = df['message'].apply(lambda x: x.count('!')).astype(int)

print("\n   Average message length:")
print(f"     Ham  : {df[df['label']=='ham']['msg_length'].mean():.0f} chars")
print(f"     Spam : {df[df['label']=='spam']['msg_length'].mean():.0f} chars")
print(f"\n   Average word count:")
print(f"     Ham  : {df[df['label']=='ham']['word_count'].mean():.1f} words")
print(f"     Spam : {df[df['label']=='spam']['word_count'].mean():.1f} words")

# =====================================================
# 3. TEXT PREPROCESSING
# =====================================================
print("\n[3] Text Preprocessing...")

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove phone numbers
    text = re.sub(r'\d{5,}', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_message'] = df['message'].apply(preprocess_text)

# Simple stopwords (no NLTK needed)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it',
    'its', 'they', 'them', 'their', 'what', 'which', 'who', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'need',
    'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'so', 'yet', 'both',
    'either', 'neither', 'not', 'only', 'own', 'same', 'than', 'too',
    'very', 's', 't', 'just', 'don', 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
    'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
    'wasn', 'weren', 'won', 'wouldn', 'in', 'on', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'through', 'during', 'before',
    'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
    'under', 'again', 'then', 'once', 'here', 'there', 'when', 'where',
    'all', 'any', 'each', 'few', 'more', 'most', 'no', 'of', 'to', 'from',
])

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in STOPWORDS])

df['clean_message'] = df['clean_message'].apply(remove_stopwords)

print("   ✓ Lowercase conversion")
print("   ✓ URL removal")
print("   ✓ Phone number removal")
print("   ✓ Punctuation removal")
print("   ✓ Stopword removal")

# =====================================================
# 4. PREPARE DATA
# =====================================================
print("\n[4] Preparing features...")

X = df['clean_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set : {len(X_train):,} messages")
print(f"   Testing set  : {len(X_test):,} messages")

# =====================================================
# 5. BUILD & TRAIN MODELS (Pipelines)
# =====================================================
print("\n[5] Training models...")

pipelines = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf',   MultinomialNB(alpha=0.1))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf',   LogisticRegression(max_iter=1000, C=5))
    ]),
    'SVM (Linear)': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf',   LinearSVC(C=1.0, max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)),
        ('clf',   RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
}

results = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {
        'pipeline': pipe, 'y_pred': y_pred,
        'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1': f1
    }

    print(f"\n   {name}:")
    print(f"     Accuracy  : {acc*100:.2f}%")
    print(f"     Precision : {prec*100:.2f}%")
    print(f"     Recall    : {rec*100:.2f}%")
    print(f"     F1 Score  : {f1*100:.2f}%")

# =====================================================
# 6. BEST MODEL
# =====================================================
best_name = max(results, key=lambda k: results[k]['F1'])
best = results[best_name]

print(f"\n{'='*60}")
print(f"  BEST MODEL : {best_name}")
print(f"  Accuracy   : {best['Accuracy']*100:.2f}%")
print(f"  Precision  : {best['Precision']*100:.2f}%")
print(f"  Recall     : {best['Recall']*100:.2f}%")
print(f"  F1 Score   : {best['F1']*100:.2f}%")
print(f"{'='*60}")

print("\n  Classification Report:")
print(classification_report(y_test, best['y_pred'], target_names=['Ham', 'Spam']))

# =====================================================
# 7. VISUALIZATIONS
# =====================================================
print("\n[6] Generating visualizations...")

fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ACCENT  = '#e8ff47'
ACCENT2 = '#ff6b35'
ACCENT3 = '#60a5fa'
BG      = '#111111'

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#888', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#333')

# Plot 1: Class Distribution
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)
counts = df['label'].value_counts()
bars = ax1.bar(['Ham ✉️', 'Spam 🚫'], counts.values,
               color=[ACCENT3, ACCENT2], edgecolor='#0a0a0a', linewidth=0.5)
for bar, count in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{count:,}', ha='center', color='white', fontsize=10, fontweight='bold')
ax1.set_title('Dataset Distribution', color='white', fontsize=12, pad=10)
ax1.set_ylabel('Count', color='#888')

# Plot 2: Message Length Distribution
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)
ham_lens  = df[df['label']=='ham']['msg_length']
spam_lens = df[df['label']=='spam']['msg_length']
ax2.hist(ham_lens,  bins=40, alpha=0.7, color=ACCENT3,  label='Ham',  density=True)
ax2.hist(spam_lens, bins=40, alpha=0.7, color=ACCENT2, label='Spam', density=True)
ax2.set_title('Message Length Distribution', color='white', fontsize=12, pad=10)
ax2.set_xlabel('Characters', color='#888')
ax2.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)

# Plot 3: Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3)
cm = confusion_matrix(y_test, best['y_pred'])
im = ax3.imshow(cm, cmap='YlOrRd')
ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Ham', 'Spam'], color='white')
ax3.set_yticklabels(['Ham', 'Spam'], color='white')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='black' if cm[i,j] > cm.max()/2 else 'white',
                 fontsize=14, fontweight='bold')
ax3.set_title(f'Confusion Matrix\n({best_name})', color='white', fontsize=12, pad=10)
ax3.set_xlabel('Predicted', color='#888')
ax3.set_ylabel('Actual', color='#888')

# Plot 4: Model Comparison — Accuracy
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4)
names  = list(results.keys())
accs   = [results[m]['Accuracy']*100 for m in names]
colors = [ACCENT if m == best_name else '#333' for m in names]
short  = ['Naive\nBayes', 'Logistic\nReg', 'SVM\nLinear', 'Random\nForest']
bars4  = ax4.bar(short, accs, color=colors, edgecolor='#222')
ax4.set_ylim(85, 102)
ax4.set_title('Accuracy Comparison', color='white', fontsize=12, pad=10)
ax4.set_ylabel('Accuracy (%)', color='#888')
for bar, acc in zip(bars4, accs):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{acc:.1f}%', ha='center', color='white', fontsize=9)

# Plot 5: F1 Score Comparison
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5)
f1s   = [results[m]['F1']*100 for m in names]
bars5 = ax5.bar(short, f1s, color=colors, edgecolor='#222')
ax5.set_ylim(70, 102)
ax5.set_title('F1 Score Comparison', color='white', fontsize=12, pad=10)
ax5.set_ylabel('F1 Score (%)', color='#888')
for bar, f1 in zip(bars5, f1s):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{f1:.1f}%', ha='center', color='white', fontsize=9)

# Plot 6: Top Spam Words
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6)
spam_text = ' '.join(df[df['label']=='spam']['clean_message'])
spam_words = Counter(spam_text.split())
top_words  = dict(sorted(spam_words.items(), key=lambda x: x[1], reverse=True)[:10])
ax6.barh(list(top_words.keys())[::-1], list(top_words.values())[::-1],
         color=ACCENT2, alpha=0.85)
ax6.set_title('Top Spam Keywords', color='white', fontsize=12, pad=10)
ax6.set_xlabel('Frequency', color='#888')

plt.suptitle('SMS Spam Classifier — NLP & Machine Learning\nBy: Ahmed Adel Shosha',
             color='white', fontsize=14, fontweight='bold', y=1.01)

plt.savefig('/mnt/user-data/outputs/spam_classifier_results.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("   Saved: spam_classifier_results.png")

# =====================================================
# 8. REAL-TIME CLASSIFIER
# =====================================================
print("\n[7] Testing real-time classifier...")

test_messages = [
    "WINNER!! You have been selected to receive a £1000 cash prize! Call 09061743182 NOW to claim!",
    "Hey, are you coming to the meeting tomorrow at 3pm?",
    "FREE entry! Win a brand new iPhone! Text WIN to 85069. £1.50/msg",
    "Can you please send me the report when you get a chance?",
    "Congratulations! You've won a 2-week holiday to Maldives! Call 0800 555 1234 to claim your FREE prize!",
    "I'll be home late tonight, don't wait for me for dinner",
]

print("\n   Real-time predictions:")
print("   " + "-"*56)
for msg in test_messages:
    pred = best['pipeline'].predict([msg])[0]
    label = "🚫 SPAM" if pred == 1 else "✅ HAM "
    print(f"   {label} | {msg[:55]}...")
print("   " + "-"*56)

print(f"\n{'='*60}")
print(f"  PROJECT COMPLETE!")
print(f"  Best Model : {best_name}")
print(f"  Accuracy   : {best['Accuracy']*100:.2f}%")
print(f"  F1 Score   : {best['F1']*100:.2f}%")
print(f"  Dataset    : 5,572 real SMS messages")
print("=" * 60)
