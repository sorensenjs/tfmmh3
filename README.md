# tfmmh3
TFMMH3 - A pure Tensorflow implementation of MurmurHash3.

## About:

This is a direct port of the public domain Pyhton implementation from 
[pymmh3](https://github.com/wc-duck/pymmh3) with most of the same caveats.

Note that because tensorflow does not have a 128 bit integer type, the interface
is changed to return tensors.

## Testing

Tested by hashing all lines in Jane Eyre by Charlotte Bronte in both c and with tfmmh3.

```
python3 -m venv venv
./venv/bin/python -m pip install "tensorflow[and-cuda]"
./venv/bin/python test/tfmmh3_test.py
```

## License

Murmur3 hash was originally created by Austin Appleby.
 +  http://code.google.com/p/smhasher/

pymmh3 was written by Fredrik Kihlander and enhanced by Swapnil Gusani, and was
placed in the public domain.  The authors hereby disclaim copyright to this source code.

tfmmh3 was adapted by Jeffrey Sorensen, and is also released in public domain,
and disclaimss any copyright.
