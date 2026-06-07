
for oo in 1 2 3 4 5
do
	for jj in topics topics2
	do
		odir=out/$oo/$jj
		mkdir -p $odir
		for ii in `ls persuader_config*`
		do
			python3 debate_ollama.py --topics $jj.json --persuader-prompt $ii --turns 3 --output $odir/$ii
		done
	done
done
