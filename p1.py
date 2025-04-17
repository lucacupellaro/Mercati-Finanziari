import time
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import numpy as np

class MyIbApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.historicalDataList = []
        self.contractDetailsList = []
        self.got_contract = False

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"Errore (reqId {reqId}, codice {errorCode}): {errorString}")

    def nextValidId(self, orderId):
        # Primo passo: cerchiamo i contratti future GC
        contract = Contract()
        contract.symbol = "GC"
        contract.secType = "FUT"
        contract.currency = "USD"
        contract.exchange = "COMEX"

        self.reqContractDetails(reqId=1, contract=contract)

    def contractDetails(self, reqId, contractDetails):
        """
        Riceviamo i dettagli dei contratti disponibili.
        """
        self.contractDetailsList.append(contractDetails)

    def contractDetailsEnd(self, reqId):
        """
        Quando abbiamo finito di ricevere i contratti.
        """
        if self.contractDetailsList:
            # Prendiamo il primo contratto disponibile (il più vicino come scadenza)
            selected_contract = self.contractDetailsList[0].contract
            print(f"Selezionato contratto: {selected_contract.symbol} {selected_contract.lastTradeDateOrContractMonth}")

            # Ora possiamo richiedere i dati storici
            self.reqHistoricalData(
                reqId=2,
                contract=selected_contract,
                endDateTime="",
                durationStr="12 M",
                barSizeSetting="1 hour",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )
        else:
            print("Nessun contratto trovato.")
            self.disconnect()

    def historicalData(self, reqId, bar):
        self.historicalDataList.append({
            'Data': bar.date,
            'Apertura': bar.open,
            'Massimo': bar.high,
            'Minimo': bar.low,
            'Prezzo': bar.close,  # Cambiato da 'Chiusura' a 'Prezzo'
            'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        print("Ricezione dati storici conclusa.")
        df = pd.DataFrame(self.historicalDataList)
        print(df.head())

        # Salva i dati grezzi
        df.to_csv("dati_storici_gc.csv", index=False)
        print("Dati salvati in 'dati_storici_gc.csv'.")

        # Calcola Volume Profile
        vp_df = MyIbApp.calcola_volume_profile_giornaliero(df)

        # Salva Volume Profile
        vp_df.to_csv("VolumeProfile.csv", index=False)
        print("Volume Profile salvato in 'VolumeProfile.csv'.")

        self.disconnect()

    def calcola_volume_profile_giornaliero(df, soglia_value_area=0.70):
        """
        Calcola, per ogni giorno nel DataFrame, POC, Value Area Low (VAL) e Value Area High (VAH).

        Argomenti:
        ----------
        df : Pandas DataFrame
            Deve contenere almeno le colonne 'Data', 'Prezzo', 'Volume'.
            La colonna 'Data' può avere date o datetime. Se c'è ora e minuto,
            verranno ignorati (si raggruppa per data).
        soglia_value_area : float
            Percentuale del volume (in forma decimale) che definisce la Value Area;
            tipicamente 0.70 (70%).

        Ritorno:
        --------
        ris : Pandas DataFrame
            Un DataFrame con le colonne:
            - 'Data'
            - 'POC'
            - 'VAL'
            - 'VAH'
        """

        # Per sicurezza, convertiamo la colonna "Data" in formato datetime (se non lo è già),
        # e prendiamo solo la parte di data per raggruppare.
        df['Data'] = pd.to_datetime(df['Data']).dt.date

        risultati = []

        # Raggruppa per la singola data
        gruppi_per_data = df.groupby('Data')

        for giorno, gruppo in gruppi_per_data:
            # Raggruppiamo i volumi per prezzo (sommando i volumi sullo stesso prezzo)
            volume_by_price = gruppo.groupby('Prezzo')['Volume'].sum().reset_index()

            # Ordiniamo per prezzo
            volume_by_price = volume_by_price.sort_values('Prezzo').reset_index(drop=True)

            # Calcoliamo il volume totale del giorno
            volume_totale = volume_by_price['Volume'].sum()
            if volume_totale == 0:
                # Se volume_totale è 0 (caso estremo), continuiamo comunque
                risultati.append({
                    'Data': giorno,
                    'POC': np.nan,
                    'VAL': np.nan,
                    'VAH': np.nan
                })
                continue

            # Troviamo il POC (prezzo con volume massimo)
            idx_poc = volume_by_price['Volume'].idxmax()
            poc_price = volume_by_price.loc[idx_poc, 'Prezzo']

            volume_by_price_sorted = volume_by_price.sort_values('Volume', ascending=False).reset_index(drop=True)

            # Spostiamo la riga corrispondente al POC in cima
            poc_row = volume_by_price_sorted[volume_by_price_sorted['Prezzo'] == poc_price]
            volume_by_price_sorted = pd.concat([
                poc_row,
                volume_by_price_sorted[volume_by_price_sorted['Prezzo'] != poc_price]
            ], ignore_index=True)

            # Ora sommiamo i volumi finché non raggiungiamo la soglia
            cumulato = 0
            prezzi_selezionati = []
            for i, row in volume_by_price_sorted.iterrows():
                cumulato += row['Volume']
                prezzi_selezionati.append(row['Prezzo'])
                if cumulato >= soglia_value_area * volume_totale:
                    break

            # Troviamo min e max dei prezzi selezionati => VAL, VAH
            val = min(prezzi_selezionati)
            vah = max(prezzi_selezionati)

            risultati.append({
                'Data': giorno,
                'POC': poc_price,
                'VAL': val,
                'VAH': vah
            })

        ris = pd.DataFrame(risultati)
        return ris


def main():
    app = MyIbApp()
    app.connect("127.0.0.1", 7497, clientId=0)
    time.sleep(1)
    app.run()


if __name__ == "__main__":
    main()