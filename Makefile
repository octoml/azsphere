CMAKE_FLAGS = -G "Ninja" \
	-DCMAKE_TOOLCHAIN_FILE="/opt/azurespheresdk/CMakeFiles/AzureSphereToolchain.cmake" \
	-DAZURE_SPHERE_TARGET_API_SET="5+Beta2004" \
	-DCMAKE_BUILD_TYPE="Debug" \
	../

build_dir := build

############################################################################
#azure sphere commands
############################################################################
program:
	azsphere device sideload deploy --imagepackage $(build_dir)/azsphere_tvm.imagepackage

start:
	azsphere device app start --componentID 1689d8b2-c835-2e27-27ad-e894d6d15fa9

stop:
	azsphere device app stop --componentID 1689d8b2-c835-2e27-27ad-e894d6d15fa9

delete_a7:
	azsphere device sideload delete --componentID 1689d8b2-c835-2e27-27ad-e894d6d15fa9

delete_m4:
	azsphere device sideload delete --componentID 18b8807b-541b-4953-8c9f-9135eedcc376

connect:
	sudo /opt/azurespheresdk/Tools/azsphere_connect.sh

restart:
	azsphere device restart

enable_development:
	azsphere device enable-development

list:
	azsphere device img list-installed

install_ethernet:
	azsphere image-package pack-board-config --preset lan-enc28j60-isu0-int5 --output enc28j60-isu0-int5.imagepackage
	azsphere device sideload deploy --imagepackage enc28j60-isu0-int5.imagepackage
	rm -f enc28j60-isu0-int5.imagepackage

remove_ethernet:
	azsphere device sideload delete --componentID 1cf5f2f3-19a8-4782-9330-3ab046b95e89

############################################################################
# build imagepackage
############################################################################
test: $(build_dir)/test_model.o
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) -DTEST=ON && ninja

conv2d: $(build_dir)/conv2d_model.o
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) -DCONV2D=ON && ninja

conv2d_network: $(build_dir)/conv2d_model.o $(build_dir)/id.bin
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) -DCONV2D_NET=ON && ninja

cifar: $(build_dir)/cifar_model.o $(build_dir)/cifar_graph.bin $(build_dir)/cifar_graph.json.c $(build_dir)/cifar_params.bin 
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) -DCIFAR=ON && ninja

keyword: $(build_dir)/keyword_model.o
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) -DKEYWORD=ON && ninja

kws_demo: $(build_dir)/keyword_model.o
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) -DDEMO=ON && ninja
############################################################################
# build model
############################################################################
$(build_dir)/test_model.o: python/build_model.py
	python3 $< -o $(build_dir) --test

$(build_dir)/conv2d_model.o: python/build_model.py
	python3 $< -o $(build_dir) --conv2d

$(build_dir)/cifar_model.o $(build_dir)/cifar_graph.bin $(build_dir)/cifar_params.bin $(build_dir)/cifar_data.bin $(build_dir)/cifar_output.bin: build_model.py
	python3 $< -o $(build_dir) --cifar
	# --tuned
	# --quantize

$(build_dir)/keyword_model.o: python/build_model.py
	python3 $< -o $(build_dir) --keyword --footprint
	# --tuned
	# --footprint

$(build_dir)/id.bin: python/build_model.py
	python3 $< -o $(build_dir) --id

clean:
	rm -rf $(build_dir)

cleanall:
	rm -rf $(build_dir)
	