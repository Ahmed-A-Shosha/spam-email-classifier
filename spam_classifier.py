"""
====================================================
  SMS Spam Classifier — NLP & Machine Learning
  By: Ahmed Adel Shosha
  Dataset: UCI SMS Spam Collection (Kaggle)
  5,572 real SMS messages
====================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re, string
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
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
print("   Dataset: UCI SMS Spam Collection (Kaggle)")
print("=" * 60)

# =====================================================
# 1. LOAD REAL DATASET
# =====================================================
print("\n[1] Loading real dataset...")

df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1])
df.columns = ['label', 'message']
df = df.dropna().reset_index(drop=True)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"   Total messages : {len(df):,}")
print(f"   Ham  (legit)   : {df['label'].value_counts()['ham']:,} ({df['label'].value_counts()['ham']/len(df)*100:.1f}%)")
print(f"   Spam           : {df['label'].value_counts()['spam']:,} ({df['label'].value_counts()['spam']/len(df)*100:.1f}%)")

# =====================================================
# 2. EDA
# =====================================================
print("\n[2] Exploratory Data Analysis...")

df['msg_length']   = df['message'].apply(len)
df['word_count']   = df['message'].apply(lambda x: len(x.split()))
df['has_currency'] = df['message'].apply(lambda x: any(c in x.lower() for c in ['£','$','€','cash','prize','win','free'])).astype(int)
df['exclamations'] = df['message'].apply(lambda x: x.count('!')).astype(int)
df['uppercase_ratio'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x),1))

print(f"   Avg Ham length  : {df[df['label']=='ham']['msg_length'].mean():.0f} chars")
print(f"   Avg Spam length : {df[df['label']=='spam']['msg_length'].mean():.0f} chars")
print(f"   Avg Ham words   : {df[df['label']=='ham']['word_count'].mean():.1f}")
print(f"   Avg Spam words  : {df[df['label']=='spam']['word_count'].mean():.1f}")

# =====================================================
# 3. TEXT PREPROCESSING
# =====================================================
print("\n[3] Text Preprocessing...")

STOPWORDS = set([
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','what','which','who','this','that',
    'these','those','am','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should',
    'a','an','the','and','but','or','so','not','just','now','very',
    'in','on','at','by','for','with','about','from','to','of','up','out',
    'all','any','each','more','no','s','t','d','ll','m','re','ve','y',
])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d{5,}', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(w for w in text.split() if w not in STOPWORDS)
    return text

df['clean'] = df['message'].apply(preprocess)

print("   ✓ Lowercase, URL removal, punctuation removal")
print("   ✓ Phone number removal, stopword removal")

# =====================================================
# 4. PREPARE DATA
# =====================================================
print("\n[4] Preparing data...")

X = df['clean']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# =====================================================
# 5. TRAIN MODELS
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
    results[name] = {
        'pipeline': pipe, 'y_pred': y_pred,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
    }
    r = results[name]
    print(f"\n   {name}:")
    print(f"     Accuracy  : {r['Accuracy']*100:.2f}%")
    print(f"     Precision : {r['Precision']*100:.2f}%")
    print(f"     Recall    : {r['Recall']*100:.2f}%")
    print(f"     F1 Score  : {r['F1']*100:.2f}%")

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
print(classification_report(y_test, best['y_pred'], target_names=['Ham','Spam']))

# =====================================================
# 7. VISUALIZATIONS
# =====================================================
print("\n[6] Generating visualizations...")

fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT  = '#e8ff47'
ACCENT2 = '#ff6b35'
ACCENT3 = '#60a5fa'
BG      = '#111111'

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#888', labelsize=9)
    for sp in ax.spines.values():
        sp.set_color('#333')

# 1 — Class Distribution
ax1 = fig.add_subplot(gs[0,0])
style_ax(ax1)
counts = df['label'].value_counts()
bars = ax1.bar(['Ham ✉️','Spam 🚫'], counts.values,
               color=[ACCENT3, ACCENT2], edgecolor='#0a0a0a')
for bar, c in zip(bars, counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
             f'{c:,}', ha='center', color='white', fontsize=10, fontweight='bold')
ax1.set_title('Dataset Distribution', color='white', fontsize=12, pad=10)
ax1.set_ylabel('Count', color='#888')

# 2 — Message Length
ax2 = fig.add_subplot(gs[0,1])
style_ax(ax2)
ax2.hist(df[df['label']=='ham']['msg_length'],  bins=50, alpha=0.7, color=ACCENT3, label='Ham',  density=True)
ax2.hist(df[df['label']=='spam']['msg_length'], bins=50, alpha=0.7, color=ACCENT2, label='Spam', density=True)
ax2.set_title('Message Length Distribution', color='white', fontsize=12, pad=10)
ax2.set_xlabel('Characters', color='#888')
ax2.legend(facecolor='#1a1a1a', labelcolor='white', fontsize=9)

# 3 — Confusion Matrix
ax3 = fig.add_subplot(gs[0,2])
style_ax(ax3)
cm = confusion_matrix(y_test, best['y_pred'])
ax3.imshow(cm, cmap='YlOrRd')
ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
ax3.set_xticklabels(['Ham','Spam'], color='white')
ax3.set_yticklabels(['Ham','Spam'], color='white')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i,j]), ha='center', va='center',
                 color='black' if cm[i,j]>cm.max()/2 else 'white',
                 fontsize=14, fontweight='bold')
ax3.set_title(f'Confusion Matrix\n({best_name})', color='white', fontsize=12, pad=10)
ax3.set_xlabel('Predicted', color='#888')
ax3.set_ylabel('Actual', color='#888')

# 4 — Accuracy Comparison
ax4 = fig.add_subplot(gs[1,0])
style_ax(ax4)
names  = list(results.keys())
short  = ['Naive\nBayes','Logistic\nReg','SVM\nLinear','Random\nForest']
accs   = [results[m]['Accuracy']*100 for m in names]
colors = [ACCENT if m==best_name else '#333' for m in names]
bars4  = ax4.bar(short, accs, color=colors, edgecolor='#222')
ax4.set_ylim(90, 101)
ax4.set_title('Accuracy Comparison', color='white', fontsize=12, pad=10)
ax4.set_ylabel('Accuracy (%)', color='#888')
for bar, acc in zip(bars4, accs):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f'{acc:.1f}%', ha='center', color='white', fontsize=9)

# 5 — F1 Comparison
ax5 = fig.add_subplot(gs[1,1])
style_ax(ax5)
f1s   = [results[m]['F1']*100 for m in names]
bars5 = ax5.bar(short, f1s, color=colors, edgecolor='#222')
ax5.set_ylim(80, 101)
ax5.set_title('F1 Score Comparison', color='white', fontsize=12, pad=10)
ax5.set_ylabel('F1 Score (%)', color='#888')
for bar, f1 in zip(bars5, f1s):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f'{f1:.1f}%', ha='center', color='white', fontsize=9)

# 6 — Top Spam Words
ax6 = fig.add_subplot(gs[1,2])
style_ax(ax6)
spam_words = Counter(' '.join(df[df['label']=='spam']['clean']).split())
top = dict(sorted(spam_words.items(), key=lambda x: x[1], reverse=True)[:10])
ax6.barh(list(top.keys())[::-1], list(top.values())[::-1], color=ACCENT2, alpha=0.85)
ax6.set_title('Top Spam Keywords', color='white', fontsize=12, pad=10)
ax6.set_xlabel('Frequency', color='#888')

plt.suptitle('SMS Spam Classifier — NLP & Machine Learning\nBy: Ahmed Adel Shosha',
             color='white', fontsize=14, fontweight='bold', y=1.01)

plt.savefig('/mnt/user-data/outputs/spam_classifier_results.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("   Saved!")

# =====================================================
# 8. REAL-TIME TEST
# =====================================================
print("\n[7] Real-time predictions...")

tests = [
    "WINNER!! You have been selected to receive a £1000 cash prize! Call 09061743182 NOW!",
    "Hey, are you coming to the meeting tomorrow at 3pm?",
    "FREE entry! Win a brand new iPhone! Text WIN to 85069. £1.50/msg",
    "Can you please send me the report when you get a chance?",
    "Congratulations! You've won a 2-week holiday! Call 0800 555 1234 to claim FREE prize!",
    "I'll be home late tonight, don't wait for me for dinner",
]

print("\n   " + "-"*56)
for msg in tests:
    pred  = best['pipeline'].predict([msg])[0]
    label = "🚫 SPAM" if pred == 1 else "✅ HAM "
    print(f"   {label} | {msg[:52]}...")
print("   " + "-"*56)

print(f"\n{'='*60}")
print(f"  PROJECT COMPLETE!")
print(f"  Best Model : {best_name}")
print(f"  Accuracy   : {best['Accuracy']*100:.2f}%")
print(f"  F1 Score   : {best['F1']*100:.2f}%")
print(f"  Dataset    : {len(df):,} real SMS messages (Kaggle UCI)")
print("=" * 60)
