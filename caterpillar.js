// # Script to auto-call Upstox API to download data of OHLCV
// NSE_EQ%7CINE848E01016

import axios from "axios"
import fs from "node:fs"

async function downloadOHLCVData() {
    const instrumentKey = 'NSE_INDEX|Nifty 50';

    const url = `https://api.upstox.com/v3/historical-candle/${instrumentKey}/minutes/5/2025-01-02/2025-01-01`;
    const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer {your_access_token}'
    };

    const startDate = new Date('2022-01-01')
    const endDate = new Date();

    let fetchedData = [];

    while(endDate > startDate) {
        let date = new Date(endDate);
        date.setDate(startDate.getDate() - 30)

        axios.get(url, {headers})
        .then(response => {
            
        })
    }
}