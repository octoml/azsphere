CMAKE_FLAGS = -G "Ninja" \
	-DCMAKE_TOOLCHAIN_FILE="/opt/azurespheresdk/CMakeFiles/AzureSphereRTCoreToolchain.cmake" \
	-DARM_GNU_PATH=${ARM_GNU_PATH} \
	-DAZURE_SPHERE_TARGET_API_SET="latest-beta" \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE \
	-DCMAKE_BUILD_TYPE="Debug" \
	../

build_dir:=build

############################################################################
#azure sphere commands
############################################################################
#flash the app on M4
program:
	azsphere device sideload deploy --imagepackage $(build_dir)/azsphere_mic.imagepackage

#start the app
start:
	azsphere device app start --componentID 18b8807b-541b-4953-8c9f-9135eedcc376

#stop the app
stop:
	azsphere device app stop --componentID 18b8807b-541b-4953-8c9f-9135eedcc376

#get astatus of the app
status:
	azsphere device app show-status --componentID 18b8807b-541b-4953-8c9f-9135eedcc376

#delete the app from M4
delete_m4:
	azsphere device sideload delete --componentID 18b8807b-541b-4953-8c9f-9135eedcc376

#delete partner app on A7
delete_a7:
	azsphere device sideload delete --componentID 1689d8b2-c835-2e27-27ad-e894d6d15fa9

#connect to azure sphere
connect:
	sudo /opt/azurespheresdk/Tools/azsphere_connect.sh

#restart device
restart:
	azsphere device restart

#list installed app on azure sphere
list:
	azsphere device img list-installed

#enable Real-Time debugging
enable_development:
	azsphere device enable-development --enablertcoredebugging

############################################################################
#build imagepackage
############################################################################
demo1: 
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) -DDEMO1=ON && ninja -v

demo2:
	@mkdir -p $(build_dir)
	cd $(build_dir) && cmake $(CMAKE_FLAGS) -DDEMO2=ON && ninja -v

clean:
	rm -rf ${build_dir}

cleanall:
	rm -rf ${build_dir}