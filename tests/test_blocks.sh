# /bin/sh

t=32
b=1

if [ $# -ge 2 ]; then

    while [ $b -lt 33 ]
    do
        ns=$($1 -t $t -b $b -m 2>&1 > /dev/null);
        echo "$b,$t,$ns" >> $2;
     
	echo "Finished test with $(($t * $b)) threads"
        #t=$(($t * 2));
	b=$(($b + 1));
    done

else
    echo "Format: ./test.sh ./[executable] [output_file]"
fi

