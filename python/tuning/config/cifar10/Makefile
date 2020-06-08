CMAKE_FLAGS = -G "Ninja" \
	-DCMAKE_TOOLCHAIN_FILE="/opt/azurespheresdk/CMakeFiles/AzureSphereToolchain.cmake" \
	-DAZURE_SPHERE_TARGET_API_SET="4" \
	-DAZURE_SPHERE_TARGET_HARDWARE_DEFINITION_DIRECTORY="../../../../Hardware/mt3620_rdb" \
	-DAZURE_SPHERE_TARGET_HARDWARE_DEFINITION="sample_hardware.json" \
	--no-warn-unused-cli \
	-DCMAKE_BUILD_TYPE="Debug" \
	-DCMAKE_MAKE_PROGRAM="ninja" \
	../

build_dir := build

delete:
	azsphere device sideload delete

connect:
	sudo /opt/azurespheresdk/Tools/azsphere_connect.sh

program:
	azsphere device sideload deploy --imagepackage $(build_dir)/octoml_AS.imagepackage
	
start:
	azsphere device app start

stop:
	azsphere device app stop

# build image package
imagepackage: $(build_dir)/conv2d_model.o $(build_dir)/conv2d_graph.bin $(build_dir)/conv2d_params.bin $(build_dir)/conv2d_data.bin $(build_dir)/conv2d_output.bin $(build_dir)/id.bin
	@mkdir -p $(@D)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) && ninja

# Serialize files
# $(build_dir)/conv2d_graph.json.c: $(build_dir)/conv2d_graph.json
# 	xxd -i $^  > $@

# $(build_dir)/conv2d_params.bin.c: $(build_dir)/conv2d_params.bin
# 	xxd -i $^  > $@

clean:
	rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake rules.ninja build.ninja

cleanall:
	rm -rf $(build_dir) *.json *.c CMakeLists.txt *.log Makefile
	