'''
pymmh3 was written by Fredrik Kihlander and enhanced by Swapnil Gusani, and is placed in the public
domain. The authors hereby disclaim copyright to this source code.

pure tensorflow implementation of the murmur3 hash algorithm

https://code.google.com/p/smhasher/wiki/MurmurHash3

This module is written to have the same format as mmh3 python package found here for simple conversions:

https://pypi.python.org/pypi/mmh3/2.3.1
'''

import tensorflow as tf


@tf.function
def _fmix32( h ):
    tf.debugging.assert_type( h, tf_type=tf.uint32 )
    h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 16))
    h = tf.math.multiply( h, 0x85ebca6b )
    h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 13))
    h = tf.math.multiply( h, 0xc2b2ae35 )
    h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 16))
    return h


@tf.function
def _fmix64( k ):
    tf.debugging.assert_type( k, tf_type=tf.uint64 )
    k = tf.bitwise.bitwise_xor( k, tf.bitwise.right_shift( k, 33 ))
    k = tf.multiply( k, 0xff51afd7ed558ccd )
    k = tf.bitwise.bitwise_xor( k, tf.bitwise.right_shift( k, 33 ))
    k = tf.multiply( k, 0xc4ceb9fe1a85ec53 )
    k = tf.bitwise.bitwise_xor( k, tf.bitwise.right_shift( k, 33 ))
    return k


@tf.function
def hash( key, seed = tf.constant( 0, tf.uint32 ) ):
    ''' Implements 32bit murmur3 hash. '''
    tf.debugging.assert_type( key, tf_type=tf.string )
    tf.debugging.assert_type( seed, tf_type=tf.uint32 )

    length = tf.strings.length( key )
    nblocks = tf.bitwise.right_shift( length, 2 )

    h1 = seed

    c1 = tf.constant( 0xcc9e2d51, tf.uint32 )
    c2 = tf.constant( 0x1b873593, tf.uint32 )

    # body
    if nblocks > 0:
        blocks = tf.cast(
            tf.io.decode_raw(
                key, tf.int32, little_endian=True, fixed_length=nblocks * 4),
            tf.uint32 )

        for i in tf.range( nblocks ):
            k1 = blocks[ i ]

            k1 = tf.math.multiply( c1, k1 )
            k1 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k1, tf.constant( 15, tf.uint32 ) ),
                 tf.bitwise.right_shift( k1, tf.constant( 17, tf.uint32 ) ) )
            k1 = tf.math.multiply( c2, k1 )

            h1 = tf.bitwise.bitwise_xor( h1, k1 )
            h1 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( h1, tf.constant( 13, tf.uint32 ) ),
                 tf.bitwise.right_shift( h1, tf.constant( 19, tf.uint32 ) ) )
            h1  = tf.math.add( tf.math.multiply( h1, 5 ), 0xe6546b64 )

    # tail
    tail_index = nblocks * 4
    k1 = tf.constant( 0, tf.uint32 )
    tail_size = length & 3
    tail_bytes = tf.cast(
        tf.io.decode_raw(tf.strings.substr( key, tail_index, tail_size ),
                         tf.uint8, little_endian=True), tf.uint32 )
    if tail_size >= 3:
        k1 = tf.bitwise.bitwise_xor(
            k1, tf.bitwise.left_shift( tail_bytes[ 2 ], 16 ) )
    if tail_size >= 2:
        k1 = tf.bitwise.bitwise_xor(
            k1, tf.bitwise.left_shift( tail_bytes[ 1 ], 8 ) )
    if tail_size >= 1:
        k1 = tf.bitwise.bitwise_xor( k1, tail_bytes[0] )

    if tail_size > 0:
        k1  = tf.math.multiply( k1, c1 )
        k1 = tf.bitwise.bitwise_or(
             tf.bitwise.left_shift( k1, tf.constant( 15, tf.uint32 ) ),
             tf.bitwise.right_shift( k1, tf.constant( 17, tf.uint32 ) ) )
        k1  = tf.math.multiply( k1, c2 )
        h1 = tf.bitwise.bitwise_xor( h1, k1 )

    #finalization
    return _fmix32( tf.bitwise.bitwise_xor( h1, tf.cast( length, tf.uint32 ) ) )


@tf.function
def hash128( key, seed = tf.constant( 0, tf.uint32 ), x64arch = True ):
    ''' Implements 128bit murmur3 hash. '''
    tf.debugging.assert_type( key, tf_type=tf.string )
    tf.debugging.assert_type( seed, tf_type=tf.uint32 )

    def hash128_x64( key, seed ):
        ''' Implements 128bit murmur3 hash for x64. '''

        length = tf.strings.length( key )
        nblocks = tf.bitwise.right_shift( length, 4 )

        h1 = tf.cast( seed, tf.uint64 )
        h2 = h1

        c1 = tf.constant( 0x87c37b91114253d5, tf.uint64 )
        c2 = tf.constant( 0x4cf5ad432745937f, tf.uint64 )

        #body
        if nblocks > 0:
            blocks = tf.cast(
                tf.io.decode_raw(
                    key, tf.int64, little_endian=True, fixed_length=nblocks * 16),
                    tf.uint64 )
            for block_start in tf.range( 0, nblocks * 2, 2 ):
                k1 = blocks[ block_start ]
                k2 = blocks[ 1 + block_start ]

                k1 = tf.math.multiply( c1, k1 )
                k1 = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( k1, tf.constant( 31, tf.uint64 ) ),
                    tf.bitwise.right_shift( k1, tf.constant( 33, tf.uint64 ) ) )
                k1 = tf.math.multiply( c2, k1 )
                h1 = tf.bitwise.bitwise_xor( h1, k1 )

                h1 = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( h1, tf.constant( 27, tf.uint64 ) ),
                    tf.bitwise.right_shift( h1, tf.constant( 37, tf.uint64 ) ) )
                h1 = tf.math.add( h1, h2 )
                h1 = tf.math.add( tf.math.multiply( h1, 5 ), 0x52dce729 )

                k2 = tf.math.multiply( c2, k2 )
                k2 = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( k2, tf.constant( 33, tf.uint64 ) ),
                    tf.bitwise.right_shift( k2, tf.constant( 31, tf.uint64 ) ) )
                k2 = tf.math.multiply( c1, k2 )
                h2 = tf.bitwise.bitwise_xor( h2, k2 )

                h2 = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( h2, tf.constant( 31, tf.uint64 ) ),
                    tf.bitwise.right_shift( h2, tf.constant( 33, tf.uint64 ) ) )
                h2 = tf.math.add( h1, h2 )
                h2 = tf.math.add( tf.math.multiply( h2, 5 ), 0x38495ab5 )

        #tail
        tail_index = nblocks * 16
        k1 = tf.constant(0, tf.uint64)
        k2 = tf.constant(0, tf.uint64)
        tail_size = length & 15
        tail_bytes = tf.cast(
            tf.io.decode_raw( tf.strings.substr( key, tail_index, tail_size ),
                              tf.uint8, little_endian=True ), tf.uint64 )

        if tail_size >= 15:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 14 ], 48 ) )
        if tail_size >= 14:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 13 ], 40 ) )
        if tail_size >= 13:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 12 ], 32 ) )
        if tail_size >= 12:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 11 ], 24 ) )
        if tail_size >= 11:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 10 ], 16 ) )
        if tail_size >= 10:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 9 ], 8 ) )
        if tail_size >=  9:
            k2 = tf.bitwise.bitwise_xor( k2, tail_bytes[ 8 ] )

        if tail_size > 8:
            k2 = tf.math.multiply( k2, c2 )
            k2 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k2, tf.constant( 33, tf.uint64 ) ),
                 tf.bitwise.right_shift( k2, tf.constant( 31, tf.uint64 ) ) )
            k2 = tf.math.multiply( k2, c1 )
            h2 = tf.bitwise.bitwise_xor( h2, k2 )

        if tail_size >=  8:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 7 ], 56 ) )
        if tail_size >=  7:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 6 ], 48 ) )
        if tail_size >=  6:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 5 ], 40 ) )
        if tail_size >=  5:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 4 ], 32 ) )
        if tail_size >=  4:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 3 ], 24 ) )
        if tail_size >=  3:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 2 ], 16 ) )
        if tail_size >=  2:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 1 ], 8 ) )
        if tail_size >=  1:
            k1 = tf.bitwise.bitwise_xor( k1, tail_bytes[ 0 ] )

        if tail_size > 0:
            k1 = tf.math.multiply( k1, c1 )
            k1 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k1, tf.constant( 31, tf.uint64 ) ),
                 tf.bitwise.right_shift( k1, tf.constant( 33, tf.uint64 ) ) )
            k1 = tf.math.multiply( k1, c2 )
            h1 = tf.bitwise.bitwise_xor( h1, k1 )

        #finalization
        h1 = tf.bitwise.bitwise_xor( h1, tf.cast( length, tf.uint64 ) )
        h2 = tf.bitwise.bitwise_xor( h2, tf.cast( length, tf.uint64 ) )

        h1  = tf.math.add( h1, h2 )
        h2  = tf.math.add( h1, h2 )

        h1  = _fmix64( h1 )
        h2  = _fmix64( h2 )

        h1  = tf.math.add( h1, h2 )
        h2  = tf.math.add( h1, h2 )

        return tf.stack([ h2, h1 ])

    def hash128_x86( key, seed ):
        ''' Implements 128bit murmur3 hash for x86. '''

        length = tf.strings.length( key )
        nblocks = tf.bitwise.right_shift( length, 4 )

        h1 = seed
        h2 = seed
        h3 = seed
        h4 = seed

        c1 = tf.constant( 0x239b961b, tf.uint32 )
        c2 = tf.constant( 0xab0e9789, tf.uint32 )
        c3 = tf.constant( 0x38b34ae5, tf.uint32 )
        c4 = tf.constant( 0xa1e38b93, tf.uint32 )

        #body
        if nblocks > 0:
            blocks = tf.cast(
                tf.io.decode_raw(
                    key, tf.int32, little_endian=True, fixed_length=nblocks * 16),
                tf.uint32)
            for block_start in tf.range( 0, nblocks * 16, 16 ):
                block_index = tf.bitwise.right_shift( block_start, 2 )

                k1 = blocks[ block_index ]
                k2 = blocks[ block_index + 1 ]
                k3 = blocks[ block_index + 2 ]
                k4 = blocks[ block_index + 3 ]

                k1  = tf.math.multiply( c1, k1 )
                k1 = tf.bitwise.bitwise_or(
                     tf.bitwise.left_shift( k1, tf.constant( 15, tf.uint32 ) ),
                     tf.bitwise.right_shift( k1, tf.constant( 17, tf.uint32 ) ) )
                k1 = tf.math.multiply( c2, k1 )
                h1 = tf.bitwise.bitwise_xor( h1,  k1 )

                h1 = tf.bitwise.bitwise_or(
                     tf.bitwise.left_shift( h1, tf.constant( 19, tf.uint32 ) ),
                     tf.bitwise.right_shift( h1, tf.constant( 13, tf.uint32 ) ) )
                h1 = tf.math.add( h1, h2 )
                h1 = tf.math.add( tf.math.multiply( h1, 5 ), 0x561ccd1b )

                k2 = tf.math.multiply( c2, k2 )
                k2 = tf.bitwise.bitwise_or(
                     tf.bitwise.left_shift( k2, tf.constant( 16, tf.uint32 ) ),
                     tf.bitwise.right_shift( k2, tf.constant( 16, tf.uint32 ) ) )
                k2 = tf.math.multiply( c3, k2 )
                h2 = tf.bitwise.bitwise_xor( h2, k2 )

                h2 = tf.bitwise.bitwise_or(
                     tf.bitwise.left_shift( h2, tf.constant( 17, tf.uint32 ) ),
                     tf.bitwise.right_shift( h2, tf.constant( 15, tf.uint32 ) ) )
                h2 = tf.math.add( h2, h3 )
                h2 = tf.math.add( tf.math.multiply( h2, 5 ), 0x0bcaa747 )

                k3  = tf.math.multiply( c3, k3 )
                k3 = tf.bitwise.bitwise_or(
                     tf.bitwise.left_shift( k3, tf.constant( 17, tf.uint32 ) ),
                     tf.bitwise.right_shift( k3, tf.constant( 15, tf.uint32 ) ) )
                k3 = tf.math.multiply( c4, k3 )
                h3 = tf.bitwise.bitwise_xor( h3, k3 )

                h3 = tf.bitwise.bitwise_or(
                     tf.bitwise.left_shift( h3, tf.constant( 15, tf.uint32 ) ),
                     tf.bitwise.right_shift( h3, tf.constant( 17, tf.uint32 ) ) )
                h3 = tf.math.add( h3, h4 )
                h3 = tf.math.add( tf.math.multiply( h3, 5 ), 0x96cd1c35 )

                k4  = tf.math.multiply( c4, k4 )
                k4 = tf.bitwise.bitwise_or(
                     tf.bitwise.left_shift( k4, tf.constant( 18, tf.uint32 ) ),
                     tf.bitwise.right_shift( k4, tf.constant( 14, tf.uint32 ) ) )
                k4 = tf.math.multiply( c1, k4 )
                h4 = tf.bitwise.bitwise_xor( h4, k4 )

                h4 = tf.bitwise.bitwise_or(
                     tf.bitwise.left_shift( h4, tf.constant( 13, tf.uint32 ) ),
                     tf.bitwise.right_shift( h4, tf.constant( 19, tf.uint32 ) ) )
                h4 = tf.math.add( h1, h4 )
                h4 = tf.math.add( tf.math.multiply( h4, 5 ),  0x32ac3b17 )

        #tail
        tail_index = nblocks * 16
        k1 = tf.constant( 0, tf.uint32 )
        k2 = k1
        k3 = k1
        k4 = k1
        tail_size = length & 15
        tail_bytes = tf.cast(
            tf.io.decode_raw( tf.strings.substr( key, tail_index, tail_size ),
                              tf.uint8, little_endian=True ), tf.uint32 )

        if tail_size >= 15:
            k4 = tf.bitwise.bitwise_xor(
                k4, tf.bitwise.left_shift(tail_bytes[ 14 ], 16))
        if tail_size >= 14:
            k4 = tf.bitwise.bitwise_xor(
                k4, tf.bitwise.left_shift(tail_bytes[ 13 ], 8))
        if tail_size >= 13:
            k4 = tf.bitwise.bitwise_xor( k4, tail_bytes[ 12 ])

        if tail_size > 12:
            k4 = tf.math.multiply( k4, c4 )
            k4 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k4, tf.constant( 18, tf.uint32 ) ),
                 tf.bitwise.right_shift( k4, tf.constant( 14, tf.uint32 ) ) )
            k4 = tf.math.multiply( k4, c1 )
            h4 = tf.bitwise.bitwise_xor( h4, k4 )

        if tail_size >= 12:
            k3 = tf.bitwise.bitwise_xor(
                k3, tf.bitwise.left_shift(tail_bytes[ 11 ], 24 ) )
        if tail_size >= 11:
            k3 = tf.bitwise.bitwise_xor(
                k3, tf.bitwise.left_shift(tail_bytes[ 10 ], 16 ) )
        if tail_size >= 10:
            k3 = tf.bitwise.bitwise_xor(
                k3, tf.bitwise.left_shift(tail_bytes[ 9 ], 8 ) )
        if tail_size >=  9:
            k3 = tf.bitwise.bitwise_xor( k3, tail_bytes[ 8 ] )

        if tail_size > 8:
            k3 = tf.math.multiply( k3, c3 )
            k3 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k3, tf.constant( 17, tf.uint32 ) ),
                 tf.bitwise.right_shift( k3, tf.constant( 15, tf.uint32 ) ) )
            k3 = tf.math.multiply( k3, c4 )
            h3 = tf.bitwise.bitwise_xor( h3, k3 )

        if tail_size >= 8:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift(tail_bytes[ 7 ], 24 ) )
        if tail_size >= 7:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift(tail_bytes[ 6 ], 16 ) )
        if tail_size >= 6:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift(tail_bytes[ 5 ], 8 ) )
        if tail_size >= 5:
            k2 = tf.bitwise.bitwise_xor( k2, tail_bytes[ 4 ] )

        if tail_size > 4:
            k2 = tf.math.multiply( k2, c2 )
            k2 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k2, tf.constant( 16, tf.uint32 ) ),
                 tf.bitwise.right_shift( k2, tf.constant( 16, tf.uint32 ) ) )
            k2 = tf.math.multiply( k2, c3 )
            h2 = tf.bitwise.bitwise_xor( h2, k2 )

        if tail_size >= 4:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift(tail_bytes[ 3 ], 24 ) )
        if tail_size >= 3:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift(tail_bytes[ 2 ], 16 ) )
        if tail_size >= 2:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift(tail_bytes[ 1 ], 8 ) )
        if tail_size >= 1:
            k1 = tf.bitwise.bitwise_xor( k1, tail_bytes[ 0 ] )

        if tail_size > 0:
            k1 = tf.math.multiply( k1, c1 )
            k1 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k1, tf.constant( 15, tf.uint32 ) ),
                 tf.bitwise.right_shift( k1, tf.constant( 17, tf.uint32 ) ) )
            k1 = tf.math.multiply( k1, c2 )
            h1 = tf.bitwise.bitwise_xor( h1, k1 )

        #finalization
        length = tf.cast( length, tf.uint32 )
        h1 = tf.bitwise.bitwise_xor( h1, length )
        h2 = tf.bitwise.bitwise_xor( h2, length )
        h3 = tf.bitwise.bitwise_xor( h3, length )
        h4 = tf.bitwise.bitwise_xor( h4, length )

        h1 = tf.math.add( h1, h2 )
        h1 = tf.math.add( h1, h3 )
        h1 = tf.math.add( h1, h4 )
        h2 = tf.math.add( h1, h2 )
        h3 = tf.math.add( h1, h3 )
        h4 = tf.math.add( h1, h4 )

        h1 = _fmix32( h1 )
        h2 = _fmix32( h2 )
        h3 = _fmix32( h3 )
        h4 = _fmix32( h4 )

        h1 = tf.math.add( h1, h2 )
        h1 = tf.math.add( h1, h3 )
        h1 = tf.math.add( h1, h4 )
        h2 = tf.math.add( h1, h2 )
        h3 = tf.math.add( h1, h3 )
        h4 = tf.math.add( h1, h4 )

        return tf.stack([
            tf.bitwise.bitwise_or(
                tf.bitwise.left_shift( tf.cast( h4, tf.uint64), 32 ),
                tf.cast( h3, tf.uint64 ) ),
            tf.bitwise.bitwise_or(
                tf.bitwise.left_shift( tf.cast( h2, tf.uint64), 32 ),
                tf.cast( h1, tf.uint64 ) )])

    if x64arch:
        return hash128_x64( key, seed )
    else:
        return hash128_x86( key, seed )


def hash64( key, seed = tf.constant( 0, tf.uint32 ), x64arch = True ):
    ''' Implements 64bit murmur3 hash. Returns a tensor of two  values. '''
    return hash128( key, seed, x64arch )


def hash_bytes( key, seed = tf.constant( 0, tf.uint32 ), x64arch = True ):
    ''' Implements 128bit murmur3 hash. Returns a byte string. '''
    tf.debugging.assert_type( key, tf_type=tf.string )
    tf.debugging.assert_type( seed, tf_type=tf.uint32 )

    hash_128 = hash128( key, seed, x64arch )

    character_lookup = tf.constant([bytes([i]) for i in range(256)])
    as_string_1 = tf.strings.reduce_join(
        tf.gather(character_lookup,
                  tf.cast(tf.bitcast(hash_128[0], tf.uint8), tf.int32)))
    as_string_2 = tf.strings.reduce_join(
        tf.gather(character_lookup,
                  tf.cast(tf.bitcast(hash_128[1], tf.uint8), tf.int32)))
    return as_string_1 + as_string_2


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser( 'pymurmur3', 'pymurmur [options] "string to hash"' )
    parser.add_argument( '--seed', type = int, default = 0 )
    parser.add_argument( 'strings', default = [], nargs='+')
    
    opts = parser.parse_args()
    
    for str_to_hash in opts.strings:
        sys.stdout.write( '"%s" = 0x%08X\n' % (
            str_to_hash, hash( tf.constant( str_to_hash, tf.string ) ).numpy() ) )
