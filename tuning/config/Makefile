CMAKE_FLAGS = -G "Ninja" \
	-DCMAKE_TOOLCHAIN_FILE="/opt/azurespheresdk/CMakeFiles/AzureSphereToolchain.cmake" \
	-DAZURE_SPHERE_TARGET_API_SET="4" \
	-DAZURE_SPHERE_TARGET_HARDWARE_DEFINITION_DIRECTORY="../../../Hardware/mt3620_rdb" \
	-DAZURE_SPHERE_TARGET_HARDWARE_DEFINITION="sample_hardware.json" \
	--no-warn-unused-cli \
	-DCMAKE_BUILD_TYPE="Debug" \
	-DCMAKE_MAKE_PROGRAM="ninja" \
	./

debug_init:
	azsphere device sideload delete

connect:
	sudo /opt/azurespheresdk/Tools/azsphere_connect.sh

# build image package
imagepackage: model.o graph.json.c params.bin.c data.bin output.bin
	@mkdir -p $(@D)
	cmake $(CMAKE_FLAGS) && ninja

# Serialize files
graph.json.c: graph.json
	xxd -i $^  > $@

params.bin.c: params.bin
	xxd -i $^  > $@

clean:
	rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake rules.ninja build.ninja

cleanall:
	rm -rf $(build_dir)
	