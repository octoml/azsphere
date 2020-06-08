CMAKE_FLAGS = -G "Ninja" \
	-DCMAKE_TOOLCHAIN_FILE="/opt/azurespheresdk/CMakeFiles/AzureSphereToolchain.cmake" \
	-DAZURE_SPHERE_TARGET_API_SET="5+Beta2004" \
	-DCMAKE_BUILD_TYPE="Debug" \
	../

build_dir := build

copy_dir := schedule_0000
task_dir := task_0002

copy:
	@mkdir -p ${build_dir}
	cp -f tuning/build/${task_dir}/${copy_dir}/build/conv2d_model.o build/
	cp -f tuning/build/${task_dir}/${copy_dir}/build/conv2d_graph.bin build/
	cp -f tuning/build/${task_dir}/${copy_dir}/build/conv2d_params.bin build/
	cp -f tuning/build/${task_dir}/${copy_dir}/build/conv2d_data.bin build/
	cp -f tuning/build/${task_dir}/${copy_dir}/build/conv2d_output.bin build/
	cp -f tuning/build/${task_dir}/${copy_dir}/build/id.bin build/

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
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

conv2d: $(build_dir)/conv2d_model.o
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

conv2d_network: $(build_dir)/conv2d_model.o $(build_dir)/id.bin
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja


#####
model: $(build_dir)/model.o $(build_dir)/graph.json.c $(build_dir)/params.bin.c

convolution: $(build_dir)/conv2d_model.o $(build_dir)/conv2d_graph.json.c $(build_dir)/conv2d_params.bin.c

# build image package
image:
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja -v

demo1: $(build_dir)/keyword_model.o $(build_dir)/keyword_graph.bin $(build_dir)/keyword_params.bin 
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

$(build_dir)/hello_imagepackage:
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja -v
	
$(build_dir)/demo_imagepackage: $(build_dir)/model.o $(build_dir)/graph.json.c $(build_dir)/params.bin.c
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

$(build_dir)/cifar_imagepackage: $(build_dir)/cifar_model.o $(build_dir)/cifar_graph.bin $(build_dir)/cifar_graph.json.c $(build_dir)/cifar_params.bin 
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

$(build_dir)/keyword_imagepackage: $(build_dir)/keyword_model.o $(build_dir)/keyword_graph.bin $(build_dir)/keyword_params.bin 
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

rebuilt:
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

$(build_dir)/sudo: $(build_dir)/conv2d_model.o $(build_dir)/conv2d_graph.bin $(build_dir)/conv2d_params.bin
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

$(build_dir)/conv2d_net_imagepackage: $(build_dir)/conv2d_model.o $(build_dir)/conv2d_graph.json.c $(build_dir)/conv2d_params.bin.c
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja



# Serialize graph.json file.
$(build_dir)/keyword_graph.json.c: $(build_dir)/keyword_graph.json
	xxd -i $^  > $@

$(build_dir)/cifar_graph.json.c: $(build_dir)/cifar_graph.json
	xxd -i $^  > $@

$(build_dir)/graph.json.c: $(build_dir)/graph.json
	xxd -i $^  > $@

$(build_dir)/conv2d_graph.json.c: $(build_dir)/conv2d_graph.json
	xxd -i $^  > $@

# Serialize params.bin file.
$(build_dir)/params.bin.c: $(build_dir)/params.bin
	xxd -i $^  > $@

$(build_dir)/conv2d_params.bin.c: $(build_dir)/conv2d_params.bin
	xxd -i $^  > $@

# build model
$(build_dir)/test_model.o: python/build_model.py
	python3 $< -o $(build_dir) --test

$(build_dir)/conv2d_model.o: python/build_model.py
	python3 $< -o $(build_dir) --conv2d

$(build_dir)/keyword_model.o $(build_dir)/keyword_graph.bin $(build_dir)/keyword_params.bin $(build_dir)/keyword_data.bin $(build_dir)/keyword_output.bin: build_model.py
	python3 $< -o $(build_dir) --keyword --footprint
	# --tuned
	# --footprint

$(build_dir)/cifar_model.o $(build_dir)/cifar_graph.bin $(build_dir)/cifar_params.bin $(build_dir)/cifar_data.bin $(build_dir)/cifar_output.bin: build_model.py
	python3 $< -o $(build_dir) --cifar
	# --tuned
	# --quantize

$(build_dir)/model.o $(build_dir)/graph.json $(build_dir)/params.bin $(build_dir)/cat.bin: build_model.py
	python3 $< -o $(build_dir) --quantize

$(build_dir)/id.bin: python/build_model.py
	python3 $< -o $(build_dir) --id

clean:
	rm -rf $(build_dir)

cleanall:
	rm -rf $(build_dir)
	