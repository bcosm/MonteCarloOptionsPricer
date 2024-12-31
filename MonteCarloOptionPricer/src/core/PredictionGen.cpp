#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <mutex> // thread-safe logging
#include <exception>
#include <memory>
#include <future>
#include <filesystem>
#include <thread> // std::this_thread::sleep_for
#include <limits> // for isfinite checks

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#include <sys/resource.h>
#include <csignal>   // for POSIX signals
#endif

// If your compiler supports OpenMP by default, include omp.h:
#ifdef _OPENMP
  #include <omp.h>
#endif

// Include your model headers
#include "../../include/models/BranchingProcessPricer.h" // ensure correct path
#include "../../include/models/MartingaleOptimizationPricer.h"
#include "../../include/models/LSMPricer.h"
#include "../../include/models/AsymptoticAnalysisPricer.h"
#include "../../include/models/RoughVolatility.h"
#include "../Grapher/Grapher.h"

// ----------------------------------------------------------------------------
// Global pointer to errorLog so signal handlers can write to it if needed
static std::shared_ptr<class SafeFileWriter> gErrorLog;

// ----------------------------------------------------------------------------
// Add safety monitoring structures
struct ProcessStats {
    std::atomic<size_t> totalMemoryUsage{0};
    std::atomic<int> activeThreads{0};
    std::atomic<int> errorCount{0};
    std::atomic<bool> shouldTerminate{false};
    
    bool isHealthy() const {
        // Check if we should terminate due to too many errors or memory usage
        return !shouldTerminate &&
               errorCount < 100000000 &&
               totalMemoryUsage < (size_t)8ULL * 1024ULL * 1024ULL * 1024ULL; // 8GB limit
    }
};

// ----------------------------------------------------------------------------
// Function to get current memory usage of this process
static size_t getCurrentMemoryUsage()
{
#ifdef _WIN32
    // Windows
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<size_t>(pmc.WorkingSetSize);
    }
    return 0;
#else
    // Linux / macOS
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // On Linux, ru_maxrss is in kilobytes; on macOS it’s in bytes.
    // We'll assume Linux by default (multiply by 1024).
    return static_cast<size_t>(usage.ru_maxrss) * 1024ULL;
#endif
}

// ----------------------------------------------------------------------------
// Safety wrapper for file operations
class SafeFileWriter {
    std::mutex mtx;
    std::ofstream file;
    std::string filepath;
    std::atomic<size_t> writeCount{0};
    static constexpr size_t FLUSH_INTERVAL = 100;

public:
    SafeFileWriter(const std::string& path) : filepath(path) {
        file.open(path, std::ios::out | std::ios::trunc);
        if (!file) throw std::runtime_error("Failed to open file: " + path);
    }

    void write(const std::string& data) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!file.good()) {
            // Attempt to reopen in append if it went bad
            file.close();
            file.open(filepath, std::ios::out | std::ios::app);
            if (!file.good()) throw std::runtime_error("File error on: " + filepath);
        }
        file << data;
        if (++writeCount % FLUSH_INTERVAL == 0) file.flush();
    }

    void writeLine(const std::string& data) {
        write(data + "\n");
    }

    ~SafeFileWriter() {
        std::lock_guard<std::mutex> lock(mtx);
        if (file.is_open()) {
            file.flush();
            file.close();
        }
    }
};

// ----------------------------------------------------------------------------
// SIGNAL HANDLERS
// ----------------------------------------------------------------------------
#ifndef _WIN32
static void signalHandler(int signum) {
    if(gErrorLog) {
        gErrorLog->writeLine("Caught signal " + std::to_string(signum) + ". Terminating process.");
    }
    std::cerr << "\nCaught signal " << signum << ". Terminating.\n";
    std::abort();
}
#endif

// --------------------------------------------------------------------
// parseDateMMDDYYYY: parse "1/3/2005" => std::tm
// --------------------------------------------------------------------
static std::tm parseDateMMDDYYYY(const std::string& dateStr) {
    std::tm tmDate = {};
    int month, day, year;
    char slash1, slash2;
    std::istringstream iss(dateStr);
    if (iss >> month >> slash1 >> day >> slash2 >> year) {
        tmDate.tm_year = year - 1900;
        tmDate.tm_mon  = month - 1;
        tmDate.tm_mday = day;
        tmDate.tm_hour = 0;
        tmDate.tm_min  = 0;
        tmDate.tm_sec  = 0;
    }
    return tmDate;
}

// Convert a std::tm to yyyymmdd (for map lookups)
static long long dateToYYYYMMDD(const std::tm& dt) {
    int year = dt.tm_year + 1900;
    int mon  = dt.tm_mon + 1;
    int day  = dt.tm_mday;
    return static_cast<long long>(year) * 10000LL
         + static_cast<long long>(mon)  * 100
         + day;
}

// ----------------------------------------------------------------------------
struct SpotData {
    // map< ticker, map< yyyymmdd, double > >
    std::unordered_map<std::string, std::unordered_map<long long, double>> hist;
};

// --------------------------------------------------------------------
// Load spot_data.csv. We assume columns: Date, AAPL, NVDA, ...
// Where Date is "1/3/2005" format, etc.
// --------------------------------------------------------------------
static void loadSpotPrices(const std::string& spotCsvPath, SpotData& outData)
{
    std::ifstream fin(spotCsvPath);
    if(!fin.is_open()) {
        std::cerr << "Cannot open " << spotCsvPath << std::endl;
        return;
    }
    std::string header;
    if(!std::getline(fin, header)) {
        std::cerr << "Empty spot_data.csv\n";
        return;
    }
    // parse columns
    std::vector<std::string> tickers;
    {
        std::istringstream hss(header);
        std::string token;
        while(std::getline(hss, token, ',')) {
            tickers.push_back(token);
        }
    }

    while(true) {
        std::string line;
        if(!std::getline(fin, line)) break;
        if(line.empty()) continue;
        std::istringstream lss(line);
        std::vector<std::string> tokens;
        {
            std::string tmp;
            while(std::getline(lss, tmp, ',')) {
                tokens.push_back(tmp);
            }
        }
        if(tokens.size() < 2) continue;

        // parse date in tokens[0], e.g. "1/3/2005"
        std::tm tmDate = parseDateMMDDYYYY(tokens[0]);
        long long yyyymmdd = dateToYYYYMMDD(tmDate);

        // read each ticker price
        for(size_t i = 1; i < tokens.size(); ++i) {
            if (i >= tickers.size()) break;
            std::string ticker = tickers[i];
            if (ticker == "Date" || ticker.empty()) continue;

            double px = 0.0;
            try {
                px = std::stod(tokens[i]);
            } catch(...) {
                // Skip invalid numeric
                continue;
            }

            // Transform ticker to lowercase
            std::transform(ticker.begin(), ticker.end(), ticker.begin(), ::tolower);
            outData.hist[ticker][yyyymmdd] = px;
        }
    }
    fin.close();
    std::cout << "Loaded spot data from " << spotCsvPath << std::endl;
}

// --------------------------------------------------------------------
// Tiered approach for how many days to fetch
// short: <= 60 => up to 10*dte
// mid:   <= 180 => up to 6*dte
// long:  >180 => up to 4*dte
// then cap at 5 yrs (~1825 days).
// --------------------------------------------------------------------
static int computeMaxDays(int dte)
{
    int factor = 10; // default
    if(dte > 60 && dte <= 180) {
        factor = 6;
    } else if(dte > 180) {
        factor = 4;
    }
    int days = factor*dte;
    if(days > 1825) days = 1825;
    return days;
}

// --------------------------------------------------------------------
// fetchSpotHistory from quoteDate backwards up to "maxDays" days
// but if data missing, we do partial
// returns oldest->newest
// --------------------------------------------------------------------
static std::vector<double> fetchSpotHistory(const SpotData& spotData,
                                            const std::string& ticker,
                                            const std::tm& quoteDate,
                                            int dte)
{
    std::vector<double> history;
    auto itTicker = spotData.hist.find(ticker);
    if(itTicker == spotData.hist.end()) {
        return history;
    }
    const auto& dailyMap = itTicker->second;

    int maxDays = computeMaxDays(dte);

    for(int back = maxDays; back >= 0; --back) {
        std::tm dt = quoteDate;
        dt.tm_mday -= back;

        // If year is <1970, skip it (avoid invalid Windows mktime range)
        if (dt.tm_year + 1900 < 1970) {
            continue;
        }

        std::time_t timeVal = std::mktime(&dt);
        if(timeVal == -1) {
            // If still invalid, skip
            continue;
        }

        long long yyyymmdd = dateToYYYYMMDD(dt);
        auto itPx = dailyMap.find(yyyymmdd);
        if(itPx != dailyMap.end()) {
            // Also skip if px is not finite
            if(!std::isfinite(itPx->second)) {
                continue;
            }
            history.push_back(itPx->second);
        }
    }
    return history; // oldest->newest
}

// --------------------------------------------------------------------
// compute20DayVolAndMomentum using 20 days => 19 log returns
// annualize stdev via sqrt(252)
// momentum = sum of 19 log returns
// if fewer than 21 points, fallback to partial or zero
// --------------------------------------------------------------------
static std::pair<double,double> compute20DayVolAndMomentum(const std::vector<double>& hist)
{
    if(hist.size() < 21) {
        return {0.0, 0.0};
    }
    // last 21 data points => 20 intervals
    std::vector<double> slice(hist.end() - 21, hist.end());
    std::vector<double> logRets; 
    logRets.reserve(20);

    for(int i = 0; i < 20; ++i) {
        double p0 = slice[i];
        double p1 = slice[i+1];
        if(p0 <= 0.0 || p1 <= 0.0) {
            logRets.push_back(0.0);
        } else {
            double lr = std::log(p1 / p0);
            if(!std::isfinite(lr)) lr = 0.0;
            logRets.push_back(lr);
        }
    }
    double sum = 0.0, sum2 = 0.0;
    for(double lr : logRets) {
        sum  += lr;
        sum2 += lr * lr;
    }
    double mean = sum / 20.0;
    double var  = (sum2 / 20.0) - (mean * mean);
    if(var < 0.0) var = 0.0;
    // annualize stdev
    double stdev    = std::sqrt(var) * std::sqrt(252.0);
    double momentum = sum; // sum of 20 log returns => ln(S_t/S_{t-20})

    return {stdev, momentum};
}

// --------------------------------------------------------------------
// Helper RAII struct to ensure activeThreads is always decremented
// even if exceptions occur
// --------------------------------------------------------------------
struct ActiveThreadGuard {
    std::atomic<int>* counter;
    explicit ActiveThreadGuard(std::atomic<int>* c) : counter(c) {
        ++(*counter);
    }
    ~ActiveThreadGuard() {
        --(*counter);
    }
};

int main()
{
    // Force unbuffered output for both stdout and stderr.
    setvbuf(stdout,  nullptr, _IONBF, 0);
    setvbuf(stderr,  nullptr, _IONBF, 0);

#ifndef _WIN32
    // Register POSIX signals
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);
    // optionally also SIGUSR1, etc.
    std::signal(SIGUSR1, signalHandler);
#endif

    // Add a custom termination handler that will catch any unhandled exception.
    std::set_terminate([] {
        std::cerr << "Uncaught fatal error. Terminating.\n";
        std::cout << "Uncaught fatal error. Terminating.\n"; // also print to console
        if (auto exc = std::current_exception()) {
            try {
                std::rethrow_exception(exc);
            } catch (const std::exception &e) {
                std::cerr << e.what() << std::endl;
                std::cout << e.what() << std::endl; // also print to console
            } catch (...) {
                std::cerr << "Unknown fatal error occurred.\n";
                std::cout << "Unknown fatal error occurred.\n";
            }
        }
        std::abort();
    });

    try {
        ProcessStats stats;
        auto errorLog = std::make_shared<SafeFileWriter>("error_log.txt");
        gErrorLog = errorLog; // so signal handlers can log
        auto resultFile = std::make_shared<SafeFileWriter>("option_data_augmented.csv");

        // Create backup of previous run if exists
        if (std::filesystem::exists("option_data_augmented.csv")) {
            try {
                std::filesystem::copy_file(
                    "option_data_augmented.csv",
                    "option_data_augmented.backup.csv",
                    std::filesystem::copy_options::overwrite_existing
                );
            } catch(...) {
                // If backup fails, we keep going
            }
        }

        // 1) Load spot data
        SpotData spotData;
        loadSpotPrices("nasdaq_stock_data.csv", spotData);

        // Write diagnostic of spot data for debugging
        {
            std::ofstream diag("spot_data_diagnostic.csv");
            if(diag.is_open()) {
                diag << "Ticker,Date,Price\n";
                for(const auto& pairTicker : spotData.hist) {
                    const auto& ticker   = pairTicker.first;
                    const auto& dailyMap = pairTicker.second;
                    for(const auto& pairDatePrice : dailyMap) {
                        diag << ticker << ","
                             << pairDatePrice.first << ","
                             << pairDatePrice.second << "\n";
                    }
                }
                diag.close();
            } else {
                std::cerr << "Failed to open spot_data_diagnostic.csv\n";
            }
        }

        // 2) Read option_data.csv
        std::ifstream fin("option_data.csv");
        if(!fin.is_open()) {
            std::cerr << "Failed to open option_data.csv.\n";
            return 1;
        }

        std::string headerLine;
        if(!std::getline(fin, headerLine)) {
            std::cerr << "Empty option_data.csv?\n";
            return 1;
        }

        // read all lines for progress
        std::vector<std::string> allLines;
        {
            std::string line;
            while(std::getline(fin, line)) {
                if(!line.empty()) {
                    allLines.push_back(line);
                }
            }
        }
        fin.close();

        int totalRows = static_cast<int>(allLines.size());
        if(totalRows == 0) {
            std::cerr << "No data lines found in option_data.csv.\n";
            return 1;
        }

        // Write header
        resultFile->write(headerLine +
            ",asymptotic_prediction"
            ",branching_prediction"
            ",lsm_prediction"
            ",martingale_prediction"
            ",twenty_day_vol"
            ",twenty_day_momentum\n");

        // A place to store final strings in row order
        std::vector<std::string> pendingResults(totalRows);
        std::vector<bool> resultReady(totalRows, false);

#ifdef _OPENMP
        std::cout << "OpenMP enabled, max threads: " << omp_get_max_threads() << "\n";
#else
        std::cout << "OpenMP not enabled\n";
#endif

        // Thread-safe logging
        std::mutex logMutex;

        // Thread-safe output ordering
        std::mutex outputMutex;
        std::atomic<int> nextRowToWrite{0};

        // Helper function to write pending results in correct order
        auto writeReadyResults = [&]() {
            std::lock_guard<std::mutex> lock(outputMutex);
            while (nextRowToWrite < totalRows && resultReady[nextRowToWrite]) {
                resultFile->writeLine(pendingResults[nextRowToWrite]);
                ++nextRowToWrite;
            }
        };

        // Periodic health check
        auto healthCheckTimer = std::async(std::launch::async, [&stats, errorLog]() {
            while (!stats.shouldTerminate) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                stats.totalMemoryUsage = getCurrentMemoryUsage(); // Update memory usage
                if (!stats.isHealthy()) {
                    errorLog->write("Process health check failed! Initiating shutdown...\n");
                    stats.shouldTerminate = true;
                    break;
                }
            }
        });

        // Keep-Alive Thread: logs every 30 seconds
        auto keepAliveTimer = std::async(std::launch::async, [&stats, &nextRowToWrite, errorLog]() {
            while(!stats.shouldTerminate) {
                std::this_thread::sleep_for(std::chrono::seconds(30));
                if(stats.shouldTerminate) break;
                errorLog->writeLine(
                    "Still alive, last row processed = " +
                    std::to_string(nextRowToWrite.load()) +
                    ", memory usage ~" + std::to_string(stats.totalMemoryUsage.load()) + " bytes.");
            }
        });

        // Global error capture
        std::atomic<bool> catastrophicFailure{false};
        std::string failureReason;
        std::mutex failureMutex;

        // Abort if we truly can't continue
        std::atomic<bool> shouldAbort{false};

        auto startTime = std::chrono::steady_clock::now(); // Track when we start

        try {
            // Parallel section
            #pragma omp parallel
            {
                try {
                    #pragma omp for schedule(dynamic)
                    for(int idx = 0; idx < totalRows; ++idx)
                    {
                        // Check for abort or health
                        #pragma omp flush(shouldAbort)
                        if (shouldAbort || stats.shouldTerminate || !stats.isHealthy() || catastrophicFailure) {
                            continue;
                        }

                        // RAII guard to ensure we always decrement activeThreads
                        ActiveThreadGuard guard(&stats.activeThreads);

                        // Extra debug
                        {
                            std::lock_guard<std::mutex> lock(logMutex);
                            errorLog->writeLine("Starting row " + std::to_string(idx));
                        }

                        const std::string line = allLines[idx];
                        try {
                            // Create local pricer instances
                            BranchingProcesses bp;
                            MartingaleOptimization mo;
                            LSM lsm;
                            AsymptoticAnalysis aa;
                            RoughVolatility roughVol;

                            std::istringstream iss(line);
                            std::vector<std::string> tokens;
                            {
                                std::string tmp;
                                while(std::getline(iss, tmp, ',')) {
                                    tokens.push_back(tmp);
                                }
                            }

                            // If not enough columns, bail with zeros
                            if(tokens.size() < 15) {
                                {
                                    std::lock_guard<std::mutex> lock(logMutex);
                                    errorLog->writeLine("Row " + std::to_string(idx) + ": Insufficient columns");
                                }
                                pendingResults[idx] = line + ",0,0,0,0,0,0";
                                resultReady[idx] = true;
                                writeReadyResults();
                                continue;
                            }

                            // parse & validate numeric inputs
                            double underlyingLast = 0.0;
                            double dteVal = 0.0;
                            double strikeDistPct = 0.0;
                            try {
                                underlyingLast   = std::stod(tokens[3]);
                                dteVal           = std::stod(tokens[4]);
                                strikeDistPct    = std::stod(tokens[5]);
                            } catch(const std::exception& e) {
                                std::lock_guard<std::mutex> lock(logMutex);
                                errorLog->writeLine(
                                    "Row " + std::to_string(idx) +
                                    ": Number parsing error: " + e.what());
                                pendingResults[idx] = line + ",0,0,0,0,0,0";
                                resultReady[idx] = true;
                                writeReadyResults();
                                ++stats.errorCount;
                                continue;
                            }

                            if (!std::isfinite(underlyingLast) ||
                                !std::isfinite(dteVal) ||
                                !std::isfinite(strikeDistPct) ||
                                underlyingLast <= 0.0 ||
                                dteVal <= 0.0 ||
                                strikeDistPct < -1.0 ||
                                strikeDistPct > 1.0)
                            {
                                // invalid numeric
                                {
                                    std::lock_guard<std::mutex> lock(logMutex);
                                    errorLog->writeLine(
                                        "Row " + std::to_string(idx) +
                                        ": Invalid numeric input(s).");
                                }
                                pendingResults[idx] = line + ",0,0,0,0,0,0";
                                resultReady[idx] = true;
                                writeReadyResults();
                                ++stats.errorCount;
                                continue;
                            }

                            // parse other columns
                            std::string ticker       = tokens[0];
                            int optionType           = 0;
                            try {
                                optionType = std::stoi(tokens[1]);
                            } catch(...) {
                                {
                                    std::lock_guard<std::mutex> lock(logMutex);
                                    errorLog->writeLine(
                                        "Row " + std::to_string(idx) +
                                        ": optionType parse error.");
                                }
                                pendingResults[idx] = line + ",0,0,0,0,0,0";
                                resultReady[idx] = true;
                                writeReadyResults();
                                ++stats.errorCount;
                                continue;
                            }

                            std::string quoteDateStr = tokens[2]; // "1/3/2005"
                            std::tm quoteDateTm      = parseDateMMDDYYYY(quoteDateStr);

                            // fetch historical data
                            int idte = static_cast<int>(dteVal);
                            std::vector<double> spotHist =
                                fetchSpotHistory(spotData, ticker, quoteDateTm, idte);

                            double asymPrice=0.0, branchPrice=0.0;
                            double lsmPriceVal=0.0, martinPrice=0.0;
                            double twentyDayVol=0.0, twentyDayMomentum=0.0;

                            // If we have any historical data at all
                            if(!spotHist.empty()) {
                                // If fewer than 2 points, push back last
                                if(spotHist.size() < 2) {
                                    spotHist.push_back(underlyingLast);
                                }
                                // Basic check that spotHist is finite
                                bool allFinite = true;
                                for(double s : spotHist) {
                                    if(!std::isfinite(s)) {
                                        allFinite = false;
                                        break;
                                    }
                                }
                                if(!allFinite) {
                                    {
                                        std::lock_guard<std::mutex> lock(logMutex);
                                        errorLog->writeLine(
                                            "Row " + std::to_string(idx) +
                                            ": Non-finite values in spotHist. Skipping.");
                                    }
                                    pendingResults[idx] = line + ",0,0,0,0,0,0";
                                    resultReady[idx] = true;
                                    writeReadyResults();
                                    ++stats.errorCount;
                                    continue;
                                }

                                // compute short-term metrics
                                auto ret = compute20DayVolAndMomentum(spotHist);
                                twentyDayVol       = ret.first;
                                twentyDayMomentum  = ret.second;

                                // define pricer params
                                double r         = 0.04;
                                double maturity  = dteVal / 365.0;
                                double dt        = 1.0 / 252.0;
                                bool isCall      = (optionType == 1);
                                double strike    = underlyingLast * (1.0 - strikeDistPct);
                                double sigma     = twentyDayVol;
                                double dividend  = 0.08; // default

                                try {
                                    dividend = std::stod(tokens[14]); // fetch dividend from column 14
                                } catch (...) {
                                    std::lock_guard<std::mutex> lock(logMutex);
                                    errorLog->writeLine(
                                        "Row " + std::to_string(idx) +
                                        ": 'dividend' parse error. Using default 0.08");
                                }

                                // Convert to int steps
                                int numTimeSteps = int(std::floor(maturity * 252.0));
                                int numPaths     = 250; 

                                if(numTimeSteps < 1) {
                                    {
                                        std::lock_guard<std::mutex> lock(logMutex);
                                        errorLog->writeLine(
                                            "Row " + std::to_string(idx) +
                                            ": No time steps => skipping pricer to avoid error.");
                                    }
                                    pendingResults[idx] = line + ",0,0,0,0,0,0";
                                    resultReady[idx] = true;
                                    writeReadyResults();
                                    ++stats.errorCount;
                                    continue;
                                }

                                // Generate MC paths
                                std::vector<std::vector<double>> pricePaths =
                                    roughVol.GenerateStockPricePaths(spotHist, numTimeSteps, numPaths);

                                if(pricePaths.empty()) {
                                    std::lock_guard<std::mutex> lock(logMutex);
                                    errorLog->writeLine(
                                        "Row " + std::to_string(idx) + ": Generated empty pricePaths. Skipping."
                                    );
                                    pendingResults[idx] = line + ",0,0,0,0,0,0";
                                    resultReady[idx] = true;
                                    writeReadyResults();
                                    ++stats.errorCount;
                                    continue;
                                }

                                // Check all paths dimensions
                                // Each path should have exactly (numTimeSteps + something) points
                                bool validPaths = true;
                                for(const auto& path : pricePaths) {
                                    if(path.size() < size_t(numTimeSteps)) {
                                        validPaths = false; 
                                        break;
                                    }
                                    for(double px : path) {
                                        if(!std::isfinite(px)) {
                                            validPaths = false; 
                                            break;
                                        }
                                    }
                                    if(!validPaths) break;
                                }
                                if(!validPaths) {
                                    std::lock_guard<std::mutex> lock(logMutex);
                                    errorLog->writeLine(
                                        "Row " + std::to_string(idx) + ": Invalid path dimension or inf/nan found."
                                    );
                                    pendingResults[idx] = line + ",0,0,0,0,0,0";
                                    resultReady[idx] = true;
                                    writeReadyResults();
                                    ++stats.errorCount;
                                    continue;
                                }

                                // define exercise times
                                std::vector<int> exerciseTimes(numTimeSteps);
                                for(int i = 0; i < numTimeSteps; ++i) {
                                    exerciseTimes[i] = i;
                                }

                                // Now do final pricer calls
                                // Catch any std::runtime_error from inside the pricers
                                try {
                                    asymPrice   = aa.PredictOptionPrice(pricePaths, r, strike, maturity, dt, isCall, sigma, dividend);
                                    branchPrice = bp.PredictOptionPrice(pricePaths, r, strike, maturity, dt, isCall, 10, exerciseTimes);
                                    lsmPriceVal = lsm.PredictOptionPrice(pricePaths, r, strike, maturity, dt, isCall, 2);
                                    martinPrice = mo.PredictOptionPrice(pricePaths, r, strike, maturity, dt, isCall, 2);
                                } catch(const std::exception& e) {
                                    {
                                        std::lock_guard<std::mutex> lock(logMutex);
                                        errorLog->writeLine(
                                            "Row " + std::to_string(idx) + 
                                            ": Exception inside pricer calls: " + e.what()
                                        );
                                    }
                                    pendingResults[idx] = line + ",0,0,0,0,0,0";
                                    resultReady[idx] = true;
                                    writeReadyResults();
                                    ++stats.errorCount;
                                    continue;
                                }
                            }

                            // build final output row
                            std::ostringstream oss;
                            oss << line << ","
                                << asymPrice        << ","
                                << branchPrice      << ","
                                << lsmPriceVal      << ","
                                << martinPrice      << ","
                                << twentyDayVol     << ","
                                << twentyDayMomentum;

                            // store & mark
                            pendingResults[idx] = oss.str();
                            resultReady[idx] = true;

                            // Write any ready lines
                            writeReadyResults();

                        } catch (const std::exception& e) {
                            // Per-row error, but not from numeric parse
                            {
                                std::lock_guard<std::mutex> lock(logMutex);
                                errorLog->writeLine(
                                    "Row " + std::to_string(idx) + 
                                    ": Unexpected error: " + e.what());
                            }
                            pendingResults[idx] = line + ",0,0,0,0,0,0";
                            resultReady[idx] = true;
                            writeReadyResults();
                            ++stats.errorCount;
                        } catch (...) {
                            {
                                std::lock_guard<std::mutex> lock(logMutex);
                                errorLog->writeLine(
                                    "Row " + std::to_string(idx) + ": Unknown error occurred");
                            }
                            pendingResults[idx] = line + ",0,0,0,0,0,0";
                            resultReady[idx] = true;
                            writeReadyResults();
                            ++stats.errorCount;
                        }

                        // Add progress tracking at the end of main loop
                        #pragma omp critical
                        {
                            auto now = std::chrono::steady_clock::now();
                            double elapsedSec = std::chrono::duration<double>(now - startTime).count();
                            double avgTimePerIter = elapsedSec / (idx + 1);
                            double timeRemain = avgTimePerIter * (totalRows - (idx + 1));
                            double progressPct = 100.0 * (idx + 1) / totalRows;
                            std::cout << "\rProgress: " << (idx + 1) << "/" << totalRows
                                      << " (" << std::fixed << std::setprecision(2) << progressPct << "%), "
                                      << "Elapsed: " << elapsedSec << "s, "
                                      << "Remain: " << timeRemain << "s, "
                                      << "Avg/iter: " << std::setprecision(3) << avgTimePerIter << "s"
                                      << std::flush;
                        }

                        // end for idx
                    }
                } catch(const std::exception& e) {
                    // Thread-level catastrophic
                    std::lock_guard<std::mutex> lock(failureMutex);
                    catastrophicFailure = true;
                    shouldAbort = true;
                    failureReason = "Thread error: " + std::string(e.what());
                    errorLog->writeLine(failureReason);
                } catch(...) {
                    // Thread-level unknown catastrophic
                    std::lock_guard<std::mutex> lock(failureMutex);
                    catastrophicFailure = true;
                    shouldAbort = true;
                    failureReason = "Unknown thread error";
                    errorLog->writeLine(failureReason);
                }
            } // end omp parallel

            if (catastrophicFailure) {
                std::cerr << "\nProcess failed: " << failureReason << std::endl;
                std::cerr << "Check error_log.txt for details\n";
                return 1;
            }

        } catch(const std::exception& e) {
            std::cerr << "\nCritical error in parallel processing: " << e.what() << std::endl;
            std::cout << "\nCritical error in parallel processing: " << e.what() << std::endl;
            return 1;
        } catch(...) {
            std::cerr << "\nUnknown critical error in parallel processing\n";
            std::cout << "\nUnknown critical error in parallel processing\n";
            return 1;
        }

        // Wait for health check timer & keep alive threads to finish
        stats.shouldTerminate = true;
        healthCheckTimer.wait();
        keepAliveTimer.wait();

        if (stats.errorCount > 0) {
            std::cout << "\nCompleted with " << stats.errorCount << " errors. Check error_log.txt\n";
        }

        // Final pass: write any remaining lines
        for(int i = nextRowToWrite.load(); i < totalRows; i++) {
            if(resultReady[i]) {
                resultFile->writeLine(pendingResults[i]);
            }
        }

        // Force a final flush of logs and result file
        errorLog.reset();
        resultFile.reset();
        gErrorLog.reset();

        std::cout << "Done. Wrote option_data_augmented.csv with new columns.\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        std::cout << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch(...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        std::cout << "Unknown fatal error occurred" << std::endl;
        return 1;
    }
}
