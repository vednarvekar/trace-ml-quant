// # Script to auto-call Upstox API to download data of OHLCV
// Reliance - NSE_EQ|INE002A01018
// HDFC - NSE_EQ|INE040A01034
// Tata Motors - NSE_EQ|INE155A01022

import axios from "axios"
import fs from "node:fs"
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv"
dotenv.config()

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const BASE_DIR = path.resolve(__dirname, ".."); 
const data_path = path.join(BASE_DIR, "data", "raw", "1min", "tataMotors_ohlcv.json");

// Ensure the directory exists
const dir = path.dirname(data_path);
if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

async function downloadOHLCVData() {
    // const instrumentKey = encodeURIComponent('NSE_INDEX|Nifty 50');
    const instrumentKey = encodeURIComponent('NSE_EQ|INE155A01022');

    const accessToken = process.env.upstoxToken;

    // Start at the beginning of the year
    let currentYear = 2022;
    let currentMonth = 0; // January
    
    const now = new Date();
    let fetchedData = [];

    console.log("--- Starting Bulletproof Extraction ---");

    while (new Date(currentYear, currentMonth, 1) < now) {
        // 1. Calculate Start and End of the calendar month
        const startOfMonth = new Date(currentYear, currentMonth, 1);
        const endOfMonth = new Date(currentYear, currentMonth + 1, 0); // Last day of current month

        const fromDateStr = startOfMonth.toISOString().split('T')[0];
        let toDate = endOfMonth > now ? now : endOfMonth;
        const toDateStr = toDate.toISOString().split('T')[0];

        const url = `https://api.upstox.com/v3/historical-candle/${instrumentKey}/minutes/1/${toDateStr}/${fromDateStr}`;

        try {
            console.log(`Fetching Month: ${fromDateStr} to ${toDateStr}`);
            const response = await axios.get(url, {
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': `Bearer ${accessToken}`
                }
            });

            if (response.data.data && response.data.data.candles) {
                const rawCandles = response.data.data.candles.reverse();
                const mappedCandles = rawCandles.map(c => ({
                    timestamp: c[0],
                    open: c[1],
                    high: c[2],
                    low: c[3],
                    close: c[4],
                    volume: c[5],
                    oi: c[6]
                }));

                fetchedData.push(...mappedCandles);
                console.log(`   Success: Added ${mappedCandles.length} candles. Total: ${fetchedData.length}`);
            }

            // 2. Increment Month
            currentMonth++;
            if (currentMonth > 11) {
                currentMonth = 0;
                currentYear++;
            }

            // 3. Respect Rate Limits
            // await new Promise(resolve => setTimeout(resolve, 1000));

        } catch (error) {
            const errorData = error.response?.data;
            console.error(`Request Failed for ${fromDateStr}:`, JSON.stringify(errorData, null, 2));
            
            if (error.response?.status === 429) {
                console.log("Rate limited! Waiting 5s...");
                await new Promise(resolve => setTimeout(resolve, 5000));
                continue; 
            }
            break; 
        }
    }

    if (fetchedData.length > 0) {
        // fs.writeFileSync('tataMotors_5min_ohlcv.json', JSON.stringify(fetchedData, null, 2));
        fs.writeFileSync(data_path, JSON.stringify(fetchedData, null, 2));
        console.log("--- ALL DATA DOWNLOADED ---");
        console.log(`Final Result: ${fetchedData.length} candles saved.`);
    }
}

downloadOHLCVData();