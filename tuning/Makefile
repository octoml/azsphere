run:
	python3 tune_azure.py --run

build:
	python3 tune_azure.py --build

plot_48:
	python3 tune_analysis.py -f server/logs/as48_poky_systemlib.txt -s npi/schedules/npi48_poky_systemlib.txt

plot_1000:
	python3 tune_analysis.py -f server/logs/as1000_armdev_gnueabihf_systemlib.txt -s npi/schedules/npi1000_armdev_gnueabihf_systemlib.txt

plot_1000_poky:
	python3 tune_analysis.py -f server/logs/as1000_armdev_poky_systemlib.txt -s npi/schedules/npi1000_armdev_poky_systemlib.txt

clean:
	rm -rf build