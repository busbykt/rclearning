python ../mavlogdump.py --types ATT --format csv firstFlight.BIN > firstFlightATT.CSV
python ../mavlogdump.py --types IMU --format csv firstFlight.BIN > firstFlightIMU.CSV
python ../mavlogdump.py --types GPS --format csv firstFlight.BIN > firstFlightGPS.CSV
python ../mavlogdump.py --types RCIN --format csv firstFlight.BIN > firstFlightRCIN.CSV
python ../mavlogdump.py --types RCOU --format csv firstFlight.BIN > firstFlightRCOU.CSV
python ../mavlogdump.py --types STAT --format csv firstFlight.BIN > firstFlightSTAT.CSV
python ../mavlogdump.py --types MODE --format csv firstFlight.BIN > firstFlightMODE.CSV
python ../mavlogdump.py --types AOA --format csv firstFlight.BIN > firstFlightAOA.CSV
python ../mavlogdump.py --types NTUN --format csv firstFlight.BIN > firstFlightNTUN.CSV