import time
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class MyIbApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.historicalDataList = []
        self.contractDetailsList = []

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"Errore (reqId {reqId}, codice {errorCode}): {errorString}")

    def nextValidId(self, orderId):
        contract = Contract()
        contract.symbol = "GC"
        contract.secType = "FUT"
        contract.currency = "USD"
        contract.exchange = "COMEX"

        self.reqContractDetails(reqId=1, contract=contract)

    def contractDetails(self, reqId, contractDetails):
        self.contractDetailsList.append(contractDetails)

    def contractDetailsEnd(self, reqId):
        if self.contractDetailsList:
            selected_contract = self.contractDetailsList[0].contract
            print(f"Selezionato contratto: {selected_contract.symbol} {selected_contract.lastTradeDateOrContractMonth}")
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
            'Prezzo': bar.close,
            'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        print("Ricezione dati storici conclusa.")
        df = pd.DataFrame(self.historicalDataList)

        df.to_csv("dati_storici_gc.csv", index=False)
        print("Dati salvati in 'dati_storici_gc.csv'.")

        vp_df = MyIbApp.calcola_volume_profile_giornaliero(df)
        vp_df.to_csv("VolumeProfile.csv", index=False)
        print("Volume Profile salvato in 'VolumeProfile.csv'.")

        # Ora lanciamo la preparazione dataset + modello
        MyIbApp.prepara_e_allena_modello(df, vp_df)

        self.disconnect()

    @staticmethod
    def calcola_volume_profile_giornaliero(df, soglia_value_area=0.70):
        df['Data'] = pd.to_datetime(df['Data']).dt.date
        risultati = []
        gruppi_per_data = df.groupby('Data')

        for giorno, gruppo in gruppi_per_data:
            volume_by_price = gruppo.groupby('Prezzo')['Volume'].sum().reset_index()
            volume_by_price = volume_by_price.sort_values('Prezzo').reset_index(drop=True)

            volume_totale = volume_by_price['Volume'].sum()
            if volume_totale == 0:
                risultati.append({'Data': giorno, 'POC': np.nan, 'VAL': np.nan, 'VAH': np.nan})
                continue

            idx_poc = volume_by_price['Volume'].idxmax()
            poc_price = volume_by_price.loc[idx_poc, 'Prezzo']

            volume_by_price_sorted = volume_by_price.sort_values('Volume', ascending=False).reset_index(drop=True)
            poc_row = volume_by_price_sorted[volume_by_price_sorted['Prezzo'] == poc_price]
            volume_by_price_sorted = pd.concat([
                poc_row,
                volume_by_price_sorted[volume_by_price_sorted['Prezzo'] != poc_price]
            ], ignore_index=True)

            cumulato = 0
            prezzi_selezionati = []
            for i, row in volume_by_price_sorted.iterrows():
                cumulato += row['Volume']
                prezzi_selezionati.append(row['Prezzo'])
                if cumulato >= soglia_value_area * volume_totale:
                    break

            val = min(prezzi_selezionati)
            vah = max(prezzi_selezionati)

            risultati.append({'Data': giorno, 'POC': poc_price, 'VAL': val, 'VAH': vah})

        return pd.DataFrame(risultati)

    @staticmethod
    def prepara_e_allena_modello(df_orario, df_vp):
        print("Preparo dataset per modello...")

        df_orario['Data'] = pd.to_datetime(df_orario['Data'])
        df_orario['Data_giorno'] = df_orario['Data'].dt.date

        aperture_giornaliere = df_orario.groupby('Data_giorno').first().reset_index()
        aperture_giornaliere = aperture_giornaliere[['Data_giorno', 'Apertura', 'Prezzo']]

        chiusure_giornaliere = df_orario.groupby('Data_giorno').last().reset_index()
        chiusure_giornaliere = chiusure_giornaliere[['Data_giorno', 'Prezzo']].rename(columns={'Prezzo': 'Chiusura'})

        df_volume_profile = df_vp.copy()
        df_volume_profile['Data_giorno'] = pd.to_datetime(df_volume_profile['Data']).dt.date

        df_merged = pd.merge(aperture_giornaliere, df_volume_profile, on='Data_giorno', how='inner')
        df_merged = pd.merge(df_merged, chiusure_giornaliere, on='Data_giorno', how='inner')

        df_merged['POC_ieri'] = df_merged['POC'].shift(1)
        df_merged['VAL_ieri'] = df_merged['VAL'].shift(1)
        df_merged['VAH_ieri'] = df_merged['VAH'].shift(1)

        df_merged['apertura_sopra_VAL_ieri'] = (df_merged['Apertura'] > df_merged['VAL_ieri']).astype(int)
        df_merged['apertura_sopra_VAH_ieri'] = (df_merged['Apertura'] > df_merged['VAH_ieri']).astype(int)

        df_merged['Chiusura_domani'] = df_merged['Chiusura'].shift(-1)
        df_merged['ritorno'] = (df_merged['Chiusura_domani'] - df_merged['Chiusura']) / df_merged['Chiusura']
        df_merged['Target'] = (df_merged['ritorno'] > 0).astype(int)

        features = ['apertura_sopra_VAL_ieri', 'apertura_sopra_VAH_ieri']
        X = df_merged[features].dropna()
        y = df_merged['Target'].dropna()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        modello = GaussianNB()
        modello.fit(X_train, y_train)

        predizioni = modello.predict(X_test)

        print("Report prestazioni modello:")
        print(classification_report(y_test, predizioni))

        ultimo_valore = X.iloc[-1:].values
        pred_oggi = modello.predict(ultimo_valore)

        if pred_oggi[0] == 1:
            print("Segnale operativo di oggi: BUY ✅")
        else:
            print("Segnale operativo di oggi: SELL ❌")


def main():
    app = MyIbApp()
    app.connect("127.0.0.1", 7497, clientId=0)
    time.sleep(1)
    app.run()

if __name__ == "__main__":
    main()
