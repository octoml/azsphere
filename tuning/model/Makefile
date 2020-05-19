build_dir:=build

# read TF model, quantize and save the module
keyword_export:
	python3 keyword_spotting.py --export --quantize --global-scale 4.0 --debug

# test accuracy on quantized model
keyword_evaluate:
	python3 keyword_spotting.py --test --quantize --module keyword_model/module_gs_4.0.pickle

clean:
	rm -rf $(build_dir)