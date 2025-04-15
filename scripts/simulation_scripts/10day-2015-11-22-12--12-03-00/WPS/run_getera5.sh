#!/bin/bash -l


DATE1=20151122
DATE2=20151203
Nort=21
West=79
Sout=-21
East=136


YY1=`echo $DATE1 | cut -c1-4`
MM1=`echo $DATE1 | cut -c5-6`
DD1=`echo $DATE1 | cut -c7-8`
YY2=`echo $DATE2 | cut -c1-4`
MM2=`echo $DATE2 | cut -c5-6`
DD2=`echo $DATE2 | cut -c7-8`


sed -e "s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;s/Nort/${Nort}/g;s/West/${West}/g;s/Sout/${Sout}/g;s/East/${East}/g;" GetERA5-sl.py > GetERA5-${DATE1}-${DATE2}-sl.py


python GetERA5-${DATE1}-${DATE2}-sl.py


sed -e "s/DATE1/${DATE1}/g;s/DATE2/${DATE2}/g;s/Nort/${Nort}/g;s/West/${West}/g;s/Sout/${Sout}/g;s/East/${East}/g;" GetERA5-pl.py > GetERA5-${DATE1}-${DATE2}-pl.py


python GetERA5-${DATE1}-${DATE2}-pl.py

