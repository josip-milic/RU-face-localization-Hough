Sveučilište u Zagrebu
Fakultet elektrotehnike i računarstva

Raspoznavanje uzoraka
http://www.fer.unizg.hr/predmet/rasuzo_a
Ak. god. 2015./2016.

++++++++++++++++++++++++++++++++++++++++++
Detekcija i lokalizacija lica na temelju 
generalizirane Houghove transformacije
++++++++++++++++++++++++++++++++++++++++++
Autori:
Mislav Larva
Tomislav Marinković
Josip Milić
Petar Pavlović
Domagoj Pereglin
Domagoj Vukadin
++++++++++++++++++++++++++++++++++++++++++
siječanj, 2016. , verzija 1.0
++++++++++++++++++++++++++++++++++++++++++


Za korištenje programske implementacije potrebno je sljedeće:
- Instalirati programski jezik Python 2.x
- Instalirati skup biblioteka SciPy (preporuka: distribucija Anaconda)
- Namjestiti putanju direktorija referentnih i ispitnih slika pomoću konstanti 
  na početku programskog koda:

--------------------------------------------------------
PROJECT_DIR  = 'C:/RU/Projekt/'
IMG_DIR_REF  = 'RU_Projekt_IMG_REF/' 
IMG_DIR      = PROJECT_DIR + 'RU_Projekt_IMG_IN/' 
IMG_SAVE_DIR = PROJECT_DIR + 'RU_Projekt_IMG_OUT/
--------------------------------------------------------
napomena: ukoliko se želi staviti putanja s uobičajenim ‘\’ umjesto ‘/’ onda 
je potrebno staviti ‘\\’ zbog t	string tipa

Potrebno je također sadržavati sljedeće direktorije sa slikama (primjeri putanja su prema gornjem primjer konstanti):
C:\RU\Projekt\ - glavni direktorij u kojem se nalaze ostali direktoriji
C:\RU\Projekt\RU_Projekt_IMG_REF - direktorij s referentnim slikama lica
C:\RU\Projekt\RU_Projekt_IMG_IN - direktorij s ispitnim slikama
C:\RU\Projekt\RU_Projekt_IMG_OUT - direktorij gdje se spremaju rezultati za pojedinu sliku
Nazivi direktorija mogu biti i drugačiji od ovih primjera, bitno je da se napomene pomoću konstanti gdje se koji direktorij 
nalazi i koje mu je ime. U direktoriju za spremanje slika se automatski nakon obrade slike stvara poddirektorij imena jednakog 
obrađenoj slici i zatim se u njega spremaju rezultati za tu ispitnu sliku.



Detalji programske implementacije se mogu pronaći u dokumentaciji projekta 
u 4. poglavlju "Opis programske implementacije rješenja".