# ComfyUI-FlowLab

[English](README.md) | **Deutsch**

Sampling-, Guidance- und Caching-Verfahren aus Papers vom Oktober 2026 als ComfyUI-Nodes. Gebaut und getestet mit **Krea 2** und **Moss-Image (Kromoss)** in ComfyUI 0.42. Alle Nodes stehen in der Kategorie `FlowLab`.

| Node | Paper | Was | Kosten | Modelle |
|---|---|---|---|---|
| Semantic Feedback Guidance (SFG) | 2610.13941 | Lenkt weg von einem Hilfszweig mit verstärktem Bild→Text-Rückfluss | +1 Modellaufruf pro Schritt im Fenster (ca. +10–15 %) | Krea 2, Moss |
| In-place SFG (iSFG) | 2610.13941 | Dämpft den Rückfluss direkt im cond-Zweig | ohne Zusatzaufruf | Krea 2, Moss |
| Trajectory-Adaptive CFG (TrajCFG) | 2610.14153 | CFG-Stärke pro Token aus der Richtungsänderung | ohne Zusatzaufruf | alle, braucht CFG > 1 |
| Edit CFG Early Correction (TI2I) | 2610.21550 | Früh im Bearbeiten: unkonditionierter Zweig → Quellbild | ohne Zusatzaufruf | Edit-Modelle mit CFG |
| SpecKCast | 2610.26837 | Cache: Kalman-Prognose pro DCT-Frequenzband | spart Aufrufe | alle |
| BeziCast | 2610.16448 | Cache: Bézier-Prognose mit Simplex-Schutz | spart Aufrufe | alle |
| CacheScope | 2610.12530 | Block-Cache mit Probe-Planung (DiCache-Rekonstruktion) | spart Blockrechnung | Krea 2, Moss |
| PoSTCache Sampler | 2610.25664 | Sampler: Anker + nachträgliche Korrektur der übersprungenen Schritte | spart Aufrufe | alle (Euler) |
| Noise-Rejuvenated SMC Sampler + Reward Model Loader | 2610.25479 | Partikel-Sampling mit Bewertungsmodell | +1 Aufruf pro Resampling, VAE-Decodes, Bewertung | alle (Euler) |
| **GRIT Realism** | eigene Methode | Zieht Frequenzbalance, Korn und Farbe der Endbild-Schätzung zur Statistik echter Fotos | praktisch keine | alle Sampler; Spektral-Teil kalibriert für Moss/Kromoss |
| FlowBudget | 2610.20636 | Schätzt die nötige Schrittzahl aus Proben echter Bilder | 2 (Euler) bzw. 4 (Heun) Aufrufe pro Probe | alle |

## Wann was hilft

**Destillierte Läufe (DMD2/Turbo, 8–16 Schritte, CFG 1):**
- **iSFG** kostet nichts und lässt den Stil unverändert.
- **SFG** im Modus `feedback (SFG-X)` stärkt Anzahl und Farbbindung.
- **NR-SMC** liefert ein besseres Einzelbild aus 4 Partikeln.
- **Caches lohnen hier nicht.**

**Basis-Läufe mit vielen Schritten und CFG:**
- **PoSTCache** ist der beste Cache.
- **TrajCFG** stabilisiert hohe CFG-Werte.

## GRIT Realism (Guided Realism via Injected Texture)

Eigene Methode für fotografische Authentizität. Messgrundlage: 60 Fotos aus DIV2K, deren Motive Gemini beschrieben hat, gegen Kromoss-Bilder derselben Motive (DMD2, 16 Schritte, CFG 1). Der KI-Look besteht messbar aus drei Teilen:
- **zu viel Mittelfrequenz-Kontrast:** +0,6 bis +1,4 dB bei 0,007–0,12 Zyklen/px
- **zu wenig feinstes Korn:** −2 dB bei 0,35 Zyklen/px, Rauschboden −30 %
- **zu viel Farbe:** Sättigung +12 %, Farbigkeit +25 %

GRIT korrigiert die geführte Endbild-Schätzung in einem σ-Fenster, und die restlichen Schritte rechnen die Änderung mit ein:
1. **realism:** ein Verstärkungsfaktor pro Latent-Kanal und DCT-Frequenzband, aus den Bildpaaren kalibriert. Moss-Latents werden dafür auf 32 Kanäle in 1/8-Auflösung entpackt.
2. **chroma:** Die Farbabweichung von Grau wird über die lineare Latent→RGB-Vorschauabbildung bestimmt, skaliert und als minimale Latent-Änderung zurückgeschrieben. Die Helligkeit bleibt dabei exakt erhalten.
3. **detail:** „Detail-Lüge“, das Modell sieht im Fenster σ·(1 − detail).

Einbau: Modell → GRIT Realism → Sampler. Es funktioniert mit KSampler, SamplerCustomAdvanced und RES4LYF. Unter `calibration` den Sampler wählen, mit dem man rechnet. Die Lücke zum echten Foto hängt vom Sampler ab: res_2s fehlt feines Detail, er_sde hat zu viel Mittelfrequenz-Kontrast. Deshalb gibt es eine Kalibrierung pro Sampler. Bei Mehrstufen-Samplern wird nur auf den Plan-Schritten korrigiert, nicht auf den Zwischenstufen.

Empfohlene Werte für Moss/Kromoss (DMD2, 16 Schritte; RES4LYF über ClownSampler mit eta 0,6, Bongmath, beta57). Der Abstand zur Statistik echter Fotos ist gemessen an 30 Motiven; kleiner ist besser.

| Sampler | calibration | realism | chroma | detail | Abstand ohne → mit GRIT |
|---|---|---|---|---|---|
| er_sde/beta | er_sde | 0,5 | 0,96 | 0,08 (Vorgabe) | 1,64 → **0,67** |
| res_2m | res_2m | 0,5 | 0,95 | 0 | 2,00 → **1,04**; ungesehene Motive: 2,18 → **1,22** |
| res_3m | res_3m | 0,5 | 0,95 (eher 0,92, Sättigung bleibt +14 %) | 0 | 1,94 → **0,94** |
| res_2s, Textur | res_2s | 0,4 | 0,95 | 0,08 | 2,32 → **1,50**; Rauschboden 0,93, Sättigung 0,97 des echten Fotos, Kontrast 0,84 |
| res_2s, ausgewogen | res_2s | 1,0 | 0,9 | 0 | 2,32 → **1,52** |

Mit er_sde/beta + GRIT liegen Rauschboden, Mikrotextur, Sättigung, Farbigkeit und Kontrast alle innerhalb von 8 % des echten Fotos. Die Gegenprobe auf 30 Motiven, die keine Kalibrierung gesehen hat, bestätigt den Gewinn (res_2m −44 %).

Was die Sampler allein zeigen:
- **er_sde/beta** liegt mit 1,64 am nächsten an echter Fotostatistik.
- **Die RES4LYF-Sampler** sind gesättigter und haben mehr Mittelfrequenz-Kontrast. res_2s ist am saubersten (feinstes Korn −3,7 dB, Rauschboden 64 %): Das wirkt hochwertig, ist aber eher Postkarte als Foto.

Sampler- und Scheduler-Vergleich (Kromoss + DMD2, 16 Schritte; RES4LYF über ClownSampler, eta 0,6, Bongmath; Abstand an 30 Motiven):

| ohne GRIT | Abstand | | mit GRIT | Abstand |
|---|---|---|---|---|
| er_sde/beta | 1,64 | | er_sde/beta | **0,67** |
| deis_3m/beta | 1,68 | | deis_3m/beta57 (calibration res_3m, 0,5/0,95) | **0,90** |
| deis_3m/beta57 | 1,72 | | res_2m/beta (res_2m, 0,5/0,95) | **0,92** |
| res_2m/beta | 1,84 | | res_2m/beta57 | 1,04 |
| abnorsett_3m/beta57 | 1,85 | | res_2m/bong_tangent (res_2m, 0,5/0,95) | 1,05 (beste Einzelstatistik, mehr Definition) |
| res_3m, res_2m, dpmpp_2m, kl_optimal, power_shift | 1,94–2,05 | | deis_3m/beta | 1,52 (beta legt zu wenig Schritte ins GRIT-Fenster) |
| ddim, bong_tangent (res_2m), res_2s, dpmpp_sde_2s, res_3s, bong_tangent (er_sde) | 2,22–2,44 | | | |

Erkenntnisse:
- **Mehr Stufen pro Schritt ergeben glattere, weniger fotografische Bilder.** res_2s, dpmpp_sde_2s und res_3s liegen hinten (Rauschboden 60–67 %).
- **Mit Shift 5 legen die meisten Scheduler 10–14 von 16 Schritten über σ = 0,7.** bong_tangent ignoriert den Shift und legt 5 Schritte unter σ = 0,3. Das ergibt mehr Definition, aber auch mehr Überschärfe: allein am schlechtesten, mit GRIT konkurrenzfähig.
- **beta schlägt beta57, kl_optimal und power_shift**, außer für GRIT mit deis_3m.

Mit Textur-LoRA und Hires-Durchgang (ein typischer Kromoss-Workflow: DMD2 + grainscape_moss, SLG Block 22, er_sde/beta, 2 MP, ×2 Lanczos mit 4 Schritten und Entrauschung 0,2; gemessen am 28.10. auf den Motiven 1–30):
- **grainscape dreht die Lücke um:** Die Mitten stimmen schon, dafür fehlt feinstes Detail (−3 bis −5 dB ab 0,2 Zyklen/px). Ohne GRIT liegt der Abstand bei 2,28, also über reinem er_sde (1,64).
- **Der Spektralteil schadet hier.** Er drückt die Mitten um etwa 1 dB zu weit herunter. Bei 1 MP:

  | Variante | Abstand |
  |---|---|
  | ohne GRIT | 2,28 |
  | realism 0,5 | 1,88 |
  | **realism 0, chroma 0,96, detail 0,08** | **1,49** |
  | nur detail 0,08 | 1,51 |
  | nur chroma | 2,13 |
  | detail 0,12 | 1,55 |

  Eine eigene Kalibrierung auf diesen Aufbau (Motive 31–60) brachte nichts (2,01). Faustregel: **Mit Textur-LoRAs realism 0.**
- **GRIT nur vor den Feinschliff schalten,** nicht in den Hires-Durchgang. Dort nimmt es die Farbe ein zweites Mal zurück. Gemessen auf 10 Motiven, verkleinert auf Referenzgröße: ohne GRIT 2,22; GRIT in beiden Durchgängen 2,09 (Farbigkeit 0,66); nur Durchgang 1 1,10. Auf 30 Motiven im vollen Workflow kommt realism 0 auf 1,51, realism 0,5 auf 1,92.

Getestet und verworfen:
- **zusätzliche späte Rauscherneuerung** (er_sde-Variante): keine messbare Wirkung, weil das destillierte Modell das Rauschen zur selben Mikrotextur zurückrechnet.
- **Detail-Lüge mit res_2m/res_3m:** verschlechtert das Ergebnis. Bei res_2s bringt sie dagegen die Textur zurück, kostet aber etwas Kontrast.
- **Eine einzige, auf er_sde kalibrierte Tabelle für alle Sampler:** macht res_2s schlechter (2,32 → 2,63), deshalb gibt es die Kalibrierung pro Sampler.

Grenzen:
- Der Spektral-Teil ist nur für Moss/Kromoss kalibriert (res_* mit 30, er_sde mit 60 Bildpaaren). chroma und detail wirken bei allen Modellen.
- Die Bildsprache des Modells (perfekte Formen, Hochglanz) ändert ein Sampler nicht.
- realism 1,0 überkorrigiert bereits, weil die Korrektur in jedem Schritt des Fensters wirkt.

## Messwerte

RTX 5070 Ti 16 GB, 1024×1024, ein Seed pro Prompt. Wenige Prompts, also eine Orientierung, kein Benchmark.

### Caches

30 Euler-Schritte, PSNR gegen den vollen Lauf, Mittel aus 3 bzw. 2 Prompts:

| Verfahren (≈12 Modellaufrufe) | Kromoss Basis, CFG 4 | Krea 2, CFG 1 |
|---|---|---|
| voller Lauf | 39 s | 44 s |
| einfach 12 Schritte | 15,3 dB | 16,7 dB |
| **PoSTCache 12** (spread) | **30,7 dB**, 16 s | **26,9 dB**, 18 s |
| BeziCast (W6, R4) | 27,8 dB | 25,5 dB |
| SpecKCast fest 12, warmup 6 | 27,9 dB | 24,9 dB |
| SpecKCast fest 12, warmup 3 (Paper) | 23,9 dB | – |
| CacheScope τ=3 | 29,4 dB bei 16 vollen Schritten, 21 s | 26,8 dB bei 15, 23 s |

Mit PoSTCache und 16 Ankern erreicht Kromoss 35,2 dB.

### SFG

Bei Kromoss + DMD2 und bei Krea 2 (Turbo-Art, CFG 1) bleibt der Stil mit **SFG-X (ω = 5)** und **iSFG** erhalten.
- SFG-X traf bei Krea 2 die Kerzenfarben (2 blau / 4 weiß) als einzige Variante.
- **SFG `both` zieht in Richtung Illustration und erzeugt bei Krea 2 Farbflecken und Rauschmuster.** Für destillierte Modelle nicht empfohlen.

### NR-SMC

Kromoss + DMD2, 16 Schritte, 4 Partikel: 59 s statt 42 s für 4 normale Bilder.
- Das beste Partikel (Index 0 bei `best first`) traf bei „fünf Enten“ und „weiße Kuh + Ballon“. Von 4 normalen Bildern traf jeweils nur jedes zweite.
- Am Ende bleiben oft zwei fast gleiche Paare übrig: Das Verfahren liefert Qualität, keine Vielfalt.

### TI2I

Nur ein Funktionstest, Kromoss ist kein Edit-Modell. Aufbau: Referenz-Latent, CFG 4, 20 Schritte, „make the raincoat bright green“.
- **Ohne Korrektur** malt CFG das Ausgangsbild um: Möwen weg, blass, körnig.
- **Mit τ = 0,1** bleibt der Aufbau erhalten, und die Bearbeitung greift trotzdem.
- PSNR zum Ausgangsbild: 16,9 → 20,7 dB (τ = 0,1) → 24,4 dB (τ = 0,3, schwächere Bearbeitung).

### FlowBudget

Kromoss Basis, CFG 4: Euler 462, Heun 111 Schritte. In der Praxis reichen 30. Mit DMD2 kommen 959 heraus, was sinnlos ist.
- Die Toleranz des Papers ist numerisch, nicht wahrnehmungsbezogen, und nur ohne CFG validiert.
- Brauchbar nur zum **Vergleichen** von Shift-, Schedule- oder Solver-Varianten untereinander.

## Verdrahtung

- **Guidance-Nodes und Caches** (MODEL → MODEL): zwischen Modell-Loader bzw. `ModelSamplingAuraFlow` und Sampler hängen. Sie lassen sich kombinieren, etwa iSFG + PoSTCache.
- **PoSTCache und NR-SMC** sind SAMPLER für `SamplerCustomAdvanced` (statt `KSamplerSelect`).
  - Für NR-SMC ist die Batch-Größe des Latents die Partikelzahl (Paper: 4).
  - `prompt` ist der Text für das Bewertungsmodell.
  - Index 0 der Ausgabe ist das beste Bild, auswählbar mit `LatentFromBatch` (Index 0, Länge 1).
- **TI2I:** `source` ist das VAE-kodierte Ausgangsbild, dasselbe wie für die Referenz-Konditionierung.
- **FlowBudget:** braucht einen GUIDER (`CFGGuider` mit Prompt und CFG), einen Batch VAE-kodierter echter Bilder und die SIGMAS des Schedulers. Das Ergebnis erscheint als INT und STRING, etwa für `PreviewAny`, und im Log.

## Abweichungen von den Papers

- **SFG:** Für Bilder nennt das Paper kein |u|; Vorgabe 0,3 (getestet wurden 0,2–0,4). Krea 2 und Moss sind Single-Stream-Modelle (Krea: Text vor Bild, Moss: Bild vor Text). Das Paper testet nur SD3/FLUX/HunyuanVideo. Qwen-Image 2.1 ist blockkausal (Text sieht nie das Bild), dort gibt es den Rückfluss nicht.
- **SpecKCast:**
  - Das Paper nennt Prozess- und Messrauschen nicht. Hier gilt die Beobachtung als exakt (R = 0). Dann kürzt sich der Prozessrausch-Maßstab aus der Vorhersage heraus, und das Update entspricht Gl. 47.
  - Die Bandleistungen p_ℓ sind für Krea 2 und Moss gemessen und eingebaut.
  - Das Risiko des adaptiven Modus wird mit Koeffizientenzahl × Bandleistung gewichtet statt mit den LPIPS-kalibrierten Gewichten des Papers. Der Modus bleibt trotzdem schlechter als der feste Plan und ist als experimentell markiert.
  - Vorgabe-Warm-up 6 statt 3, das ist gemessen besser.
- **BeziCast:** sagt die geführte Geschwindigkeit voraus statt die Repräsentation vor der letzten Schicht, und läuft damit modellunabhängig. Für u_max und die Schutzregel (ρ, γ, β_max) nennt das Paper keine Zahlen; die Vorgaben sind gesetzt, nicht getunt.
- **CacheScope:** Die Rekonstruktion ist r = r₂ + γ(r₂ − r₁) mit γ = clip(d(y, y₂)/d(y₂, y₁), 1, 1,5); die genaue DiCache-Regel für γ ist nicht veröffentlicht. Die Basis-Politik besteht aus Warm-up-Anteil + letztem Schritt. Wird cond+uncond gemeinsam gerechnet, teilen sie sich die Entscheidung. Bei ausgelagerten Modellen (Krea 2 mit 16 GB) laden übersprungene Blöcke trotzdem ihre Gewichte.
- **PoSTCache:** Vorschlag „letzten Wert halten“ (Paper-Standard). Das dichte Präfix ist die Hälfte des Budgets. Die PVR-Kalibrierung ist umgesetzt (Modus `calibrate`, Profile unter `ComfyUI/user/flowlab/postcache`), brachte bei Kromoss aber nichts gegenüber `spread`.
- **NR-SMC:** Bewertungsmodell PickScore statt ImageReward; das ImageReward-Paket pinnt alte Abhängigkeiten. Der Look-ahead nutzt den regulären nächsten Schritt und kostet deshalb keinen zusätzlichen Modellaufruf. Einen Aufruf pro Resampling für den erneuerten Zustand gibt es immer.
- **FlowBudget:** finite Differenzen statt JVPs. Die Proben-σ werden aus dem tatsächlichen Schedule gezogen.

## Voraussetzungen

- Für NR-SMC ein CLIP-Präferenzmodell im Hugging-Face-Format in `models/reward_models`, z. B. `yuvalkirstain/PickScore_v1`, dazu `transformers` (in ComfyUI vorhanden).
- Die übrigen Nodes brauchen nichts zusätzlich.

## Lizenz

GPL-3.0, wie ComfyUI selbst (siehe `LICENSE`).
