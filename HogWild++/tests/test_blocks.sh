# /bin/sh

t=32
b=1
first_ns=0

if [ $# -ge 2 ]; then

    echo "Blocks,TPB,Accuracy,Kernel Time, Malloc Time,Total Time,Kernel Speedup" >> $2;
    while [ $b -lt 100 ]
    do
        data=$($1 -t $t -b $b -m 2>&1 > /dev/null);
        ns=$(echo $data | cut -d "," -f2)
        if [ $b -eq 1 ]; then
            first_ns=$ns
        fi

        speedup=$(echo "$first_ns / $ns" | bc -l);
        echo "$b,$t,$data,$speedup" >> $2;
     
	    echo "Finished test with $(($t * $b)) threads"
        #t=$(($t * 2));
	    b=$(($b + 1));
    done

else
    echo "Format: ./test_blocks.sh ./[executable] [output_file]"
fi

