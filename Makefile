CMAKE_FLAGS = -G "Ninja" \
	-DCMAKE_TOOLCHAIN_FILE="/opt/azurespheresdk/CMakeFiles/AzureSphereToolchain.cmake" \
	-DAZURE_SPHERE_TARGET_API_SET="4" \
	-DAZURE_SPHERE_TARGET_HARDWARE_DEFINITION_DIRECTORY="./Hardware/mt3620_rdb" \
	-DAZURE_SPHERE_TARGET_HARDWARE_DEFINITION="sample_hardware.json" \
	--no-warn-unused-cli \
	-DCMAKE_BUILD_TYPE="Debug" \
	-DCMAKE_MAKE_PROGRAM="ninja" \
	~/azure-sphere

build_dir := build

debug_init:
	# rm -rf $(build_dir)
	azsphere device sideload delete

connect:
	sudo /opt/azurespheresdk/Tools/azsphere_connect.sh

model: $(build_dir)/model.o $(build_dir)/graph.json.c $(build_dir)/params.bin.c

convolution: $(build_dir)/conv2d_model.o $(build_dir)/conv2d_graph.json.c $(build_dir)/conv2d_params.bin.c

# build image package
$(build_dir)/demo_imagepackage: $(build_dir)/model.o $(build_dir)/graph.json.c $(build_dir)/params.bin.c
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

$(build_dir)/conv2d_imagepackage: $(build_dir)/conv2d_model.o $(build_dir)/conv2d_graph.json.c $(build_dir)/conv2d_params.bin.c
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

$(build_dir)/test_imagepackage: $(build_dir)/test_model.o
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

# Serialize graph.json file.
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
$(build_dir)/model.o $(build_dir)/graph.json $(build_dir)/params.bin $(build_dir)/cat.bin: build_model.py
	python3 $< -o $(build_dir) --quantize

$(build_dir)/conv2d_model.o $(build_dir)/conv2d_graph.json $(build_dir)/conv2d_params.bin $(build_dir)/conv2d_data.bin: build_model.py
	python3 $< -o $(build_dir) --conv2d

$(build_dir)/test_model.o $(build_dir)/test_graph.json $(build_dir)/test_params.bin $(build_dir)/test_data.bin $(build_dir)/test_output.bin: build_model.py
	python3 $< -o $(build_dir) --test

clean:
	rm -rf $(build_dir)

cleanall:
	rm -rf $(build_dir)
	