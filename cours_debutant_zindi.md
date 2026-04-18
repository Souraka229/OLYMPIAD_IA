# Cours pour Débutants — Intelligence Artificielle & Computer Vision
### Tout comprendre depuis zéro pour la compétition Zindi Bénin

---

## Table des matières

1. [C'est quoi l'Intelligence Artificielle ?](#ia)
2. [C'est quoi une image pour un ordinateur ?](#image)
3. [C'est quoi le Machine Learning ?](#ml)
4. [C'est quoi un réseau de neurones ?](#nn)
5. [C'est quoi le Deep Learning ?](#dl)
6. [C'est quoi PyTorch ?](#pytorch)
7. [Les données : charger et préparer les images](#donnees)
8. [Le modèle : EfficientNet expliqué simplement](#modele)
9. [L'entraînement : comment le modèle apprend](#entrainement)
10. [La validation : est-ce que le modèle est bon ?](#validation)
11. [Les prédictions et la soumission](#prediction)
12. [Les erreurs courantes et comment les éviter](#erreurs)

---

## 1. C'est quoi l'Intelligence Artificielle ? {#ia}

L'Intelligence Artificielle (IA), c'est faire en sorte qu'un ordinateur **apprenne à résoudre des problèmes** comme le ferait un humain.

### Une analogie simple

Imagine que tu veux apprendre à un enfant à reconnaître un chien. Tu lui montres des milliers de photos en disant :

- "Ça, c'est un chien" ✅
- "Ça, ce n'est pas un chien" ❌

Après avoir vu assez d'exemples, l'enfant peut reconnaître un chien qu'il n'a jamais vu.

**L'IA, c'est exactement pareil**, mais avec un ordinateur et des mathématiques.

### Notre problème dans la compétition

On montre à l'ordinateur des photos satellites en lui disant :
- "Cette image contient une route" → Label = **1**
- "Cette image ne contient pas de route" → Label = **0**

L'ordinateur apprend, et ensuite il peut regarder de nouvelles photos et dire si elles contiennent une route ou pas.

---

## 2. C'est quoi une image pour un ordinateur ? {#image}

### Les pixels

Une image, c'est un tableau de petits carrés colorés appelés **pixels**.

Une image de 224×224 pixels, c'est :
- 224 lignes
- 224 colonnes
- = **50 176 pixels** au total

### Les couleurs (RGB)

Chaque pixel a **3 valeurs** : Rouge (R), Vert (G), Bleu (B).

Chaque valeur va de **0** (sombre) à **255** (brillant).

```
Pixel rouge vif  = (255, 0, 0)
Pixel vert vif   = (0, 255, 0)
Pixel bleu vif   = (0, 0, 255)
Pixel blanc      = (255, 255, 255)
Pixel noir       = (0, 0, 0)
```

### Une image = un tableau de chiffres

Pour l'ordinateur, une image couleur de 224×224 c'est un tableau de chiffres de forme :

```
(224 lignes) × (224 colonnes) × (3 couleurs) = 150 528 chiffres
```

C'est avec ces chiffres que l'IA va travailler.

```python
from PIL import Image
import numpy as np

# Ouvrir une image
img = Image.open('ma_photo.tif').convert('RGB')

# La convertir en tableau de chiffres
tableau = np.array(img)
print(tableau.shape)  # (224, 224, 3)

# Le pixel en haut à gauche
print(tableau[0, 0])  # Ex: [120, 85, 60] → Rouge=120, Vert=85, Bleu=60
```

---

## 3. C'est quoi le Machine Learning ? {#ml}

### La différence avec la programmation classique

**Programmation classique :**
```
Règles + Données → Résultats
```
Tu écris toi-même les règles : "Si l'image a beaucoup de gris et des lignes droites, c'est une route".

**Machine Learning :**
```
Données + Résultats → Règles (trouvées automatiquement)
```
Tu donnes des exemples à l'ordinateur, et il trouve lui-même les règles.

### Pourquoi le Machine Learning est meilleur ?

Essaie d'écrire des règles pour reconnaître une route :
- Parfois les routes sont grises, parfois brunes, parfois noires
- Parfois il y a de la végétation sur les côtés, parfois non
- Les routes peuvent être larges ou étroites
- Les rivières sèches ressemblent à des routes...

C'est impossible d'écrire toutes les règles à la main ! Le Machine Learning trouve ces règles automatiquement en regardant des milliers d'exemples.

### Les 3 étapes du Machine Learning

```
1. ENTRAÎNEMENT  : montrer des exemples au modèle
                   → il apprend les règles

2. VALIDATION    : tester sur des exemples qu'il n'a jamais vus
                   → on mesure si les règles sont bonnes

3. PRÉDICTION    : utiliser le modèle sur de nouvelles images
                   → il donne sa réponse
```

---

## 4. C'est quoi un réseau de neurones ? {#nn}

### Les neurones biologiques

Dans ton cerveau, il y a ~86 milliards de neurones. Chaque neurone reçoit des signaux, les traite, et envoie un signal à d'autres neurones.

### Les neurones artificiels

Un neurone artificiel fait la même chose :

```
Entrées × Poids → Somme → Activation → Sortie
```

Exemple concret :
- Entrée 1 = Valeur rouge du pixel = 120
- Entrée 2 = Valeur verte du pixel = 85
- Poids 1 = 0.3, Poids 2 = 0.7
- Somme = (120 × 0.3) + (85 × 0.7) = 36 + 59.5 = 95.5
- Sortie = 95.5 (envoyée au prochain neurone)

### Les "poids" : c'est ce qui s'apprend

Les poids sont les paramètres du modèle. Au début, ils sont aléatoires. Pendant l'entraînement, ils sont ajustés pour faire de meilleures prédictions.

C'est ça, "apprendre" : ajuster les poids pour réduire les erreurs.

### Un réseau de neurones = plusieurs couches

```
Couche d'entrée  →  Couche cachée 1  →  Couche cachée 2  →  Couche de sortie
(pixels de       →  (motifs simples) →  (formes)         →  (route / pas route)
 l'image)
```

---

## 5. C'est quoi le Deep Learning ? {#dl}

Le Deep Learning, c'est du Machine Learning avec des réseaux de neurones qui ont **beaucoup de couches** (d'où "deep" = profond).

### Pourquoi beaucoup de couches ?

Chaque couche apprend des choses de plus en plus complexes :

```
Couche 1 : apprend à détecter des bords (lignes horizontales, verticales)
Couche 2 : combine les bords pour détecter des coins, des coins
Couche 3 : combine les formes pour détecter des textures (bitume, gravier)
Couche 4 : combine les textures pour détecter des routes, des rivières
...
Couche N : décide : "C'est une route" ou "Ce n'est pas une route"
```

### Les CNN (Convolutional Neural Networks)

Pour les images, on utilise un type spécial de réseau appelé **CNN** (Réseau de Neurones Convolutif).

La particularité : il utilise des **filtres** qui glissent sur l'image pour détecter des motifs.

Imagine un filtre "détecteur de lignes horizontales" :
```
Filtre :          Il glisse sur l'image...
[-1, -1, -1]      et partout où il trouve
[ 0,  0,  0]      une ligne horizontale,
[ 1,  1,  1]      il s'active fortement.
```

Un modèle CNN a des centaines de ces filtres qui détectent chacun quelque chose de différent.

---

## 6. C'est quoi PyTorch ? {#pytorch}

PyTorch est une bibliothèque Python créée par Facebook pour faire du Deep Learning. C'est comme une boîte à outils qui contient déjà tout ce dont on a besoin.

### Ce que PyTorch nous donne

- Les **tenseurs** : des tableaux de chiffres optimisés pour le GPU
- Les **couches de neurones** : Conv2d, Linear, ReLU, etc.
- Les **optimiseurs** : Adam, SGD pour ajuster les poids
- Les **fonctions de perte** : pour mesurer les erreurs
- Des **modèles pré-entraînés** : EfficientNet, ResNet, etc.

### C'est quoi un tenseur ?

Un tenseur, c'est simplement un tableau de chiffres. La différence avec NumPy : il peut être placé sur le GPU pour des calculs ultra-rapides.

```python
import torch

# Un tenseur simple (liste de chiffres)
t = torch.tensor([1.0, 2.0, 3.0])
print(t)        # tensor([1., 2., 3.])
print(t.shape)  # torch.Size([3])

# Un tenseur 2D (tableau)
t2 = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0]])
print(t2.shape)  # torch.Size([2, 2])

# Une image = tenseur 3D
# (3 canaux couleurs, 224 lignes, 224 colonnes)
image = torch.zeros(3, 224, 224)
print(image.shape)  # torch.Size([3, 224, 224])

# Envoyer sur le GPU
device = torch.device("cuda")  # GPU
t = t.to(device)               # Maintenant sur le GPU !
```

### GPU vs CPU — Quelle différence ?

| | CPU | GPU |
|--|-----|-----|
| Nb de cœurs | 4 à 16 | Milliers |
| Spécialité | Tâches variées | Calculs en parallèle |
| Entraînement IA | Très lent | Très rapide |
| Exemple | 30 min par epoch | 1 min par epoch |

**Toujours utiliser le GPU sur Colab !**

```python
# Vérifier si le GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # "cuda" = GPU actif ✅, "cpu" = problème ❌
```

---

## 7. Les données : charger et préparer les images {#donnees}

### Étape 1 : Lire le CSV

Le fichier `Train.csv` contient deux colonnes :
- `Image_ID` : le nom de l'image (ex: `ID_ABC123`)
- `Target` : 1 si route, 0 si pas route

```python
import pandas as pd

train_df = pd.read_csv('Train.csv')
print(train_df.head())
# Image_ID       Target
# ID_D9ONL553    1
# ID_263YTILY    0
# ID_XK9P2MN1    1
# ...
```

### Étape 2 : Diviser les données

On divise nos données en deux parties :

```
Toutes les données (7000 images)
          ↓
  ┌───────────────────────────────┐
  │  Train (80%) = 5600 images   │  ← Le modèle apprend dessus
  │  Val   (20%) = 1400 images   │  ← On mesure la qualité
  └───────────────────────────────┘
```

```python
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,          # 20% pour la validation
    stratify=train_df['Target'],  # Garder la même proportion de 0 et 1
    random_state=42         # Pour avoir le même résultat à chaque fois
)

print(f"Train : {len(train_data)} images")  # 5600
print(f"Val   : {len(val_data)} images")    # 1400
```

### Étape 3 : Les transformations d'images

Avant de donner une image au modèle, on doit la préparer :

**1. Redimensionner** : toutes les images à la même taille (224×224)
**2. Convertir en tenseur** : transformer les pixels (0-255) en chiffres (0.0-1.0)
**3. Normaliser** : centrer les valeurs autour de 0

```python
from torchvision import transforms

# Pourquoi normaliser avec ces valeurs précises ?
# Parce que EfficientNet a été entraîné sur ImageNet avec ces valeurs.
# Si on ne normalise pas de la même façon, il ne "comprend" plus les images.
MEAN = [0.485, 0.456, 0.406]  # Moyenne de chaque canal sur ImageNet
STD  = [0.229, 0.224, 0.225]  # Écart-type de chaque canal sur ImageNet

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 1. Redimensionner
    transforms.ToTensor(),          # 2. Pixels → tenseur (divise par 255)
    transforms.Normalize(MEAN, STD) # 3. Centrer les valeurs
])
```

### Étape 4 : La classe Dataset

PyTorch utilise une classe `Dataset` pour savoir comment lire les images.

C'est comme une recette : "Pour avoir l'image n°42, voilà comment la lire."

```python
from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class RoadDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        # __init__ : s'exécute quand on crée le dataset
        # C'est comme "préparer les ingrédients"
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        # __len__ : combien d'images dans le dataset ?
        return len(self.df)

    def __getitem__(self, idx):
        # __getitem__ : donne-moi l'image numéro idx
        # C'est comme "chercher la photo n°42 dans l'album"

        # 1. Trouver le nom et le label de l'image
        img_id = self.df.loc[idx, 'Image_ID']
        label  = self.df.loc[idx, 'Target']

        # 2. Construire le chemin complet
        img_path = os.path.join(self.img_dir, img_id + '.tif')

        # 3. Ouvrir l'image
        image = Image.open(img_path).convert('RGB')

        # 4. Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        # 5. Retourner l'image et son label
        return image, torch.tensor(label, dtype=torch.float32)
```

### Étape 5 : Le DataLoader

Le `DataLoader` charge les images par groupes (appelés **batches**) et les mélange.

```python
from torch.utils.data import DataLoader

# Pourquoi des batches et pas image par image ?
# Parce que le GPU est plus efficace quand il traite plusieurs images à la fois.
# batch_size=32 : traiter 32 images à la fois

train_dataset = RoadDataset(train_data, img_dir, transform)
val_dataset   = RoadDataset(val_data,   img_dir, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,    # 32 images par batch
    shuffle=True      # Mélanger à chaque epoch (évite de mémoriser l'ordre)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False     # Pas besoin de mélanger pour la validation
)

# Vérifier un batch
images, labels = next(iter(train_loader))
print(images.shape)  # torch.Size([32, 3, 224, 224])
#                                   ↑   ↑    ↑   ↑
#                               32 imgs 3 canaux 224×224 pixels
print(labels.shape)  # torch.Size([32])
```

---

## 8. Le modèle : EfficientNet expliqué simplement {#modele}

### C'est quoi EfficientNet ?

EfficientNet est un CNN très puissant créé par des ingénieurs de Google en 2019. Au lieu de l'inventer nous-mêmes, on l'utilise directement.

C'est comme utiliser une voiture déjà construite au lieu de fabriquer une voiture depuis zéro.

### Le Transfer Learning — apprendre sur les épaules des géants

EfficientNet a déjà été entraîné sur **ImageNet** : 1.2 millions d'images, 1000 catégories différentes (chats, voitures, fleurs, avions...).

Cela signifie qu'il sait déjà :
- Détecter des bords et des lignes
- Reconnaître des textures (herbe, béton, eau)
- Identifier des formes et des objets

On n'a plus qu'à lui apprendre la dernière étape : "Est-ce que parmi tout ce que tu vois, il y a une route ?"

```python
# Avant le transfer learning :
#    EfficientNet → 1000 sorties (chien, chat, voiture, fleur...)

# Après le transfer learning :
#    EfficientNet → 1 sortie (route / pas route)
```

### En code :

```python
from torchvision import models
import torch.nn as nn

# 1. Charger EfficientNet avec ses poids pré-entraînés
model = models.efficientnet_b4(weights='DEFAULT')
# weights='DEFAULT' = télécharger les poids appris sur ImageNet

# 2. Voir la dernière couche
print(model.classifier)
# Sequential(
#   (0): Dropout(p=0.4)
#   (1): Linear(in_features=1792, out_features=1000)  ← 1000 sorties
# )

# 3. Remplacer la dernière couche par une couche avec 1 seule sortie
# in_features = 1792 (nombre de neurones qui entrent dans la dernière couche)
model.classifier[1] = nn.Linear(1792, 1)  # 1 seule sortie : route ou pas

# 4. Envoyer le modèle sur le GPU
model = model.to(device)
print(f"Modèle prêt sur : {device}")
```

### Pourquoi "B4" dans EfficientNet-B4 ?

EfficientNet existe en plusieurs tailles : B0, B1, B2, B3, B4, B5, B6, B7.

Plus le chiffre est grand :
- Plus le modèle est précis ✅
- Plus il est lent à entraîner ⚠️
- Plus il consomme de mémoire GPU ⚠️

Pour la compétition, **B4** est le meilleur compromis sur Colab.

---

## 9. L'entraînement : comment le modèle apprend {#entrainement}

### Le principe en 4 étapes

Pour chaque batch d'images, le modèle fait 4 choses :

```
1. FORWARD  : regarder les images et faire une prédiction
2. LOSS     : calculer l'erreur (à quel point il s'est trompé ?)
3. BACKWARD : calculer comment corriger les poids
4. UPDATE   : corriger les poids
```

C'est comme un élève qui :
1. Répond à une question d'examen
2. Voit le corrigé et mesure son erreur
3. Comprend pourquoi il s'est trompé
4. Retient la bonne réponse pour la prochaine fois

### La fonction de perte (Loss)

La perte mesure à quel point le modèle s'est trompé. Plus la perte est basse, mieux c'est.

On utilise `BCEWithLogitsLoss` (Binary Cross-Entropy) pour la classification binaire.

```python
criterion = nn.BCEWithLogitsLoss()

# Exemple :
# Le modèle prédit  : [0.9]  (90% de chance que ce soit une route)
# La vraie réponse  : [1.0]  (c'est bien une route)
# Perte             : faible (bonne prédiction)

# Le modèle prédit  : [0.1]  (10% de chance)
# La vraie réponse  : [1.0]  (c'est une route)
# Perte             : élevée (mauvaise prédiction)
```

### L'optimiseur Adam

L'optimiseur décide comment ajuster les poids après avoir calculé les gradients.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# lr = learning rate = taux d'apprentissage
# lr=1e-4 = 0.0001
```

**Le Learning Rate : ni trop grand, ni trop petit**

```
LR trop grand (ex: 0.1) :
  Le modèle fait de trop grands sauts, il n'apprend pas
  → La loss monte et descend de façon chaotique

LR trop petit (ex: 0.000001) :
  Le modèle apprend, mais très très lentement
  → Des heures pour converger

LR bien réglé (ex: 0.0001) :
  Le modèle apprend progressivement et converge ✅
```

### La boucle complète

```python
from tqdm import tqdm  # Barre de progression

num_epochs = 10  # Nombre de fois qu'on passe sur tout le dataset

for epoch in range(num_epochs):

    # ===== PHASE 1 : ENTRAÎNEMENT =====
    model.train()  # Dire au modèle qu'il est en mode entraînement
    running_loss = 0.0

    # tqdm affiche une barre de progression
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

        # 1. Envoyer sur le GPU
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        # unsqueeze(1) transforme [32] en [32, 1] pour correspondre aux sorties du modèle

        # 2. Forward : calculer les prédictions
        outputs = model(images)       # Shape: [32, 1]

        # 3. Calculer la perte
        loss = criterion(outputs, labels)

        # 4. Remettre les gradients à zéro (OBLIGATOIRE avant backward)
        optimizer.zero_grad()

        # 5. Backward : calculer les gradients
        loss.backward()

        # 6. Update : ajuster les poids
        optimizer.step()

        # Accumuler la perte pour affichage
        running_loss += loss.item() * images.size(0)

    # Perte moyenne sur l'epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f}")
```

### C'est quoi une "epoch" ?

Une epoch = le modèle a vu **toutes les images** d'entraînement une fois.

Avec 5600 images et des batches de 32 :
- 5600 ÷ 32 = **175 batches** par epoch
- Chaque epoch = 175 mises à jour des poids

Après 10 epochs, le modèle a vu chaque image 10 fois.

---

## 10. La validation : est-ce que le modèle est bon ? {#validation}

### Pourquoi valider ?

Le modèle pourrait mémoriser les réponses du dataset d'entraînement (comme un élève qui mémorise les réponses sans comprendre). C'est ce qu'on appelle le **surapprentissage** (overfitting).

La validation utilise des images que le modèle **n'a jamais vues**. Si le score sur la validation est bon, c'est que le modèle a vraiment appris à généraliser.

```
Si : score train = 0.99, score val = 0.70
→ SURAPPRENTISSAGE : le modèle a mémorisé, pas appris

Si : score train = 0.94, score val = 0.93
→ BON APPRENTISSAGE : le modèle a vraiment appris ✅
```

### La métrique AUC

L'AUC (Area Under the Curve) va de 0 à 1. C'est la métrique officielle de la compétition.

```
AUC = 0.50 : modèle aléatoire (comme lancer une pièce)
AUC = 0.90 : très bon modèle
AUC = 0.95 : excellent modèle
AUC = 1.00 : modèle parfait (impossible en pratique)
```

```python
from sklearn.metrics import roc_auc_score

# Phase de validation
model.eval()  # Mode évaluation (désactive le dropout)
val_preds  = []  # Probabilités prédites
val_labels = []  # Vraies réponses

with torch.no_grad():  # Pas besoin de calculer les gradients
    for images, labels in val_loader:
        images = images.to(device)

        # Prédictions du modèle (logits)
        outputs = model(images)

        # Convertir en probabilités (entre 0 et 1)
        probs = torch.sigmoid(outputs)

        # Sauvegarder
        val_preds.extend(probs.cpu().numpy().flatten())
        val_labels.extend(labels.numpy().flatten())

# Calculer l'AUC
auc = roc_auc_score(val_labels, val_preds)
print(f"AUC : {auc:.4f}")
```

### C'est quoi sigmoid ?

Le modèle retourne des "logits" (des chiffres bruts, peuvent être négatifs ou très grands).
On applique sigmoid pour les transformer en probabilités entre 0 et 1.

```
sigmoid(-5)  = 0.007  → très peu probable que ce soit une route
sigmoid(-1)  = 0.27   → peu probable
sigmoid( 0)  = 0.50   → incertain (50/50)
sigmoid( 1)  = 0.73   → probable
sigmoid( 5)  = 0.993  → très probable que ce soit une route
```

---

## 11. Les prédictions et la soumission {#prediction}

### Faire les prédictions sur les images de test

Les images de test n'ont pas de label. Le modèle doit deviner.

```python
# Dataset test (sans labels)
class RoadDatasetTest(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id   = self.df.loc[idx, 'Image_ID']
        img_path = os.path.join(self.img_dir, img_id + '.tif')
        image    = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_id  # Retourner l'ID à la place du label

test_dataset = RoadDatasetTest(test_df, img_dir, transform)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Prédictions
model.eval()
test_ids   = []
test_preds = []

with torch.no_grad():
    for images, ids in test_loader:
        outputs = model(images.to(device))
        probs   = torch.sigmoid(outputs).cpu().numpy()
        test_preds.extend(probs.flatten())
        test_ids.extend(ids)

print(f"Prédictions : {len(test_preds)} images")
```

### Créer le fichier de soumission

```python
import pandas as pd

# Créer le DataFrame de soumission
submission = pd.DataFrame({
    'Image_ID': test_ids,
    'Target'  : test_preds
})

# Vérifier le format
print(submission.head())
# Image_ID       Target
# ID_D9ONL553    0.87
# ID_263YTILY    0.13
# ...

print(f"Shape : {submission.shape}")
print(f"Min : {submission['Target'].min():.3f}")
print(f"Max : {submission['Target'].max():.3f}")

# Sauvegarder
submission.to_csv('submission.csv', index=False)
print("✅ Fichier sauvegardé !")
```

**Important** : les valeurs Target doivent être des probabilités (entre 0 et 1), **pas** des 0 et 1 entiers. C'est ce qui permet à Zindi de calculer un bon score AUC.

---

## 12. Les erreurs courantes et comment les éviter {#erreurs}

### Erreur 1 : Tourner sur CPU au lieu du GPU

**Symptôme** : L'entraînement dure des heures au lieu de quelques minutes.

**Diagnostic** :
```python
print(torch.cuda.is_available())  # False → problème !
```

**Solution** : Exécution → Modifier le type d'exécution → T4 GPU

---

### Erreur 2 : Oublier `.to(device)`

**Symptôme** : `RuntimeError: Expected all tensors to be on the same device`

**Solution** :
```python
# TOUJOURS envoyer les images et les labels sur le GPU
images = images.to(device)
labels = labels.to(device)
model  = model.to(device)
```

---

### Erreur 3 : Oublier `optimizer.zero_grad()`

**Symptôme** : La loss ne diminue pas correctement.

**Pourquoi** : PyTorch accumule les gradients. Sans `zero_grad()`, chaque backward s'ajoute aux précédents.

**Solution** : Toujours appeler `optimizer.zero_grad()` **avant** `loss.backward()`.

---

### Erreur 4 : Oublier `model.eval()` pendant la validation

**Symptôme** : Les prédictions de validation sont moins bonnes qu'elles devraient être.

**Pourquoi** : Certaines couches (Dropout, BatchNorm) se comportent différemment en entraînement et en évaluation.

**Solution** :
```python
model.train()  # Pendant l'entraînement
model.eval()   # Pendant la validation et la prédiction
```

---

### Erreur 5 : Retourner des 0/1 au lieu de probabilités

**Symptôme** : Score AUC très bas (proche de 0.5).

**Solution** :
```python
# ❌ MAUVAIS : retourner des classes binaires
preds = (torch.sigmoid(outputs) > 0.5).float()  # [0, 1, 1, 0, ...]

# ✅ BON : retourner des probabilités
preds = torch.sigmoid(outputs)  # [0.87, 0.13, 0.73, 0.21, ...]
```

---

### Erreur 6 : La session Colab qui se déconnecte

**Symptôme** : Colab se ferme après 90 minutes sans activité.

**Solution** : Sauvegarder le modèle régulièrement sur Google Drive.
```python
# Sauvegarder à chaque fois qu'on améliore l'AUC
if auc > best_auc:
    best_auc = auc
    torch.save(model.state_dict(), '/content/drive/MyDrive/best_model.pth')
    print(f"✅ Modèle sauvegardé ! AUC : {auc:.4f}")
```

---

## Glossaire — Les mots importants

| Mot | Définition simple |
|-----|------------------|
| **Dataset** | L'ensemble de toutes les images et leurs labels |
| **Batch** | Un petit groupe d'images traité en une seule fois (ex: 32) |
| **Epoch** | Une passe complète sur toutes les images d'entraînement |
| **Loss** | L'erreur du modèle. Plus c'est bas, mieux c'est |
| **AUC** | La métrique de la compétition. Plus c'est proche de 1, mieux c'est |
| **GPU** | Processeur ultra-rapide pour les calculs IA |
| **Tenseur** | Tableau de chiffres utilisé par PyTorch |
| **Poids** | Les paramètres internes du modèle qui s'ajustent pendant l'apprentissage |
| **Gradient** | La direction dans laquelle corriger les poids |
| **Learning rate** | La taille des corrections à chaque étape |
| **Sigmoid** | Fonction qui transforme n'importe quel chiffre en probabilité (0 à 1) |
| **Overfitting** | Quand le modèle mémorise au lieu d'apprendre |
| **Transfer learning** | Réutiliser un modèle déjà entraîné sur un autre problème |
| **Fine-tuning** | Adapter un modèle pré-entraîné à notre problème |
| **EfficientNet** | Le modèle CNN qu'on utilise pour la compétition |
| **ImageNet** | Un grand dataset de 1.2 million d'images sur lequel EfficientNet a été formé |
| **Normalisation** | Mettre les valeurs des pixels dans un intervalle standard |
| **Data augmentation** | Créer de nouvelles images en retournant/pivotant les existantes |
| **TTA** | Test Time Augmentation : faire plusieurs prédictions et les moyenner |

---

*Ce cours t'a préparé pour comprendre et améliorer ton code pour la compétition Zindi. Bonne chance !*
