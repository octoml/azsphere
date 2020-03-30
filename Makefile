build_dir := build

debug_init:
	rm -rf $(build_dir)
	azsphere device sideload delete

connect:
	sudo /opt/azurespheresdk/Tools/azsphere_connect.sh

# build image package
$(build_dir)/imagepackage: $(build_dir)/test_model.o
	@mkdir -p $(@D)
	cd $(build_dir) && cmake \
	-G "Ninja" \
	-DCMAKE_TOOLCHAIN_FILE="/opt/azurespheresdk/CMakeFiles/AzureSphereToolchain.cmake" \
	-DAZURE_SPHERE_TARGET_API_SET="4" \
	-DAZURE_SPHERE_TARGET_HARDWARE_DEFINITION_DIRECTORY="./Hardware/mt3620_rdb" \
	-DAZURE_SPHERE_TARGET_HARDWARE_DEFINITION="sample_hardware.json" \
	--no-warn-unused-cli \
	-DCMAKE_BUILD_TYPE="Debug" \
	-DCMAKE_MAKE_PROGRAM="ninja" \
	~/azure-sphere && ninja

# building models
$(build_dir)/model.o $(build_dir)/graph.json $(build_dir)/params.bin $(build_dir)/cat.bin: build_model.py
	python3 $< -o $(build_dir)

$(build_dir)/test_model.o $(build_dir)/test_graph.json $(build_dir)/test_params.bin $(build_dir)/test_data.bin $(build_dir)/test_output.bin: build_model.py
	python3 $< -o $(build_dir) --test

clean:
	rm -rf $(build_dir)

cleanall:
	rm -rf $(build_dir)
	