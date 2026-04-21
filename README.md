# Vatnafraedi-Lokaverkefni-Hopur-13
Þetta repository inniheldur Python-kóða sem notaður er í lokaverkefni í vatnafræði var 2026.  
Markmið verkefnisins er að greina vatnafræðileg gögn fyrir valið vatnasvið og tengja þau við efni áfangans.

## Innihald repos

- `scripts/vatnafraedi_lokaverkefni.py`  
  Aðalskrá með úrvinnslu gagna og myndgerð.

- `figures/`  
  Hér vistast myndir og töflur sem kóðinn býr til.

## Gögn

Kóðinn notar eftirfarandi gagnaskrár en er hannaður til þess að geta notað gögn frá hvaða vatnasviði sem er á Íslandi:

- `Vedurgogn_ID_66.csv`
- `Rennslisgogn_ID_66.csv`
- `Eiginleikar Vatnasviðs Catchment_attributes.csv`

Gögnin eru ekki geymd í þessu repository.  
Til að keyra kóðann þarf að sækja sér gögn frá LamaH-Ice gagnasett (Helgason og Nijssen, 2024), mikilvægt að nota nýjustu útgáfu, v1.5,
sjá hlekk: https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/

Á vefsíðunni skal hlaða niður zip skránni lamah_ice.zip og pdf skjali sem lýsir
veðurbreytum. Veðurgögn eru í möppunni
"lamah_ice\A_basins_total_upstrm\2_timeseries\daily\meteorological_data" og
rennslisgögn í möppunni “lamah_ice\lamah_ice\D_gauges\2_timeseries\daily”.
Eiginleikar vatnasviða eru hér:
"\lamah_ice\A_basins_total_upstrm\1_attributes\Catchment_attributes.csv"`.

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
