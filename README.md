# Vatnafraedi-Lokaverkefni-Hopur-13

Þetta repository inniheldur Python-kóða sem notaður er í lokaverkefni í vatnafræði árið 2026.  
Markmið verkefnisins er að greina vatnafræðileg gögn fyrir valið vatnasvið og tengja niðurstöður við helstu hugtök áfangans.

## Innihald repository

- `scripts/vatnafraedi_lokaverkefni.py`  
  Aðalskrá sem framkvæmir alla úrvinnslu gagna og býr til myndir og töflur.

- `figures/`  
  Hér vistast allar myndir og sumar niðurstöður sem kóðinn býr til.

## Hvað kóðinn gerir

Kóðinn framkvæmir eftirfarandi greiningar:

1. Tekur saman helstu eiginleika vatnasviðs
2. Reiknar og teiknar árstíðarsveiflu (climatology)
3. Framkvæmir baseflow separation með Ladson-aðferð
4. Setur upp vatnsjöfnu (úrkoma, afrennsli og leif)
5. Reiknar langæislínu rennslis (flow duration curve)
6. Framkvæmir flóðagreiningu með annual peak flows og líkindadreifingum
7. Framkvæmir leitnigreiningu
8. Greinir staka rennslisatburði

## Gögn

Kóðinn notar eftirfarandi gagnaskrár:

- `Vedurgogn_ID_66.csv`
- `Rennslisgogn_ID_66.csv`
- `Eiginleikar Vatnasviðs Catchment_attributes.csv`

Kóðinn er hannaður þannig að hægt sé að nota gögn frá hvaða vatnasviði sem er, svo lengi sem uppsetning gagna og dálka er sú sama.

Gögnin eru **ekki geymd í þessu repository**.  
Til að keyra kóðann þarf að sækja gögn úr LamaH-Ice gagnasafninu:

Helgason og Nijssen (2024), útgáfa v1.5  
https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/

## Uppsetning gagna

Eftir að búið er að hlaða niður `lamah_ice.zip`:

- Veðurgögn: lamah_ice/A_basins_total_upstrm/2_timeseries/daily/meteorological_data
- Rennslisgögn: lamah_ice/D_gauges/2_timeseries/daily
- Eiginleikar vatnasviða: lamah_ice/A_basins_total_upstrm/1_attributes/Catchment_attributes.csv

  
Setja þarf viðeigandi skrár í möppuna `data/` í repository-inu áður en kóðinn er keyrður.

## Hvernig á að keyra kóðann

1. Klóna eða sækja repository-ið.
2. Búa til möppuna `data/` í rót reposins.
3. Setja gagnaskrárnar í `data/`.
4. Setja upp nauðsynlega pakka:

 - numpy  
 - pandas  
 - matplotlib  
 - scipy  

5. Keyra skrána:

```bash
python scripts/vatnafraedi_lokaverkefni.py
