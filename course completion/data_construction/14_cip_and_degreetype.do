forvalues y=2014/2019 {
	foreach v in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "D:\\Yifeng -- Project Work\\ys8mz_sandbox\\Build\\Student\\Student_`y'_`v'.dta.zip", clear
			keep vccsid strm cip intended_degreetype total_credit_hrs
			duplicates drop
			gen degree_level = 4
			replace degree_level = 1 if intended_degreetype == "College Transfer"
			replace degree_level = 2 if intended_degreetype == "Occ/Tech"
			replace degree_level = 3 if intended_degreetype == "Certificate"
			gsort vccsid strm degree_level -total_credit_hrs
			bys vccsid strm: keep if _n == 1
			drop intended_degreetype total_credit_hrs
			replace cip = floor(cip)
			tempfile tmp
			save "`tmp'", replace
		restore
		append using "`tmp'", force
	}
}
replace cip = 0 if cip == .
sort vccsid strm
merge 1:1 vccsid strm using "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\all_vccsid_strm.dta", keep(2 3) nogen
drop first_strm
bys cip: gen cip_counts = _N
replace cip = 99 if cip_counts < ceil(_N * 0.005)
sort vccsid strm
order vccsid strm
drop cip_counts
replace degree_level = 4 if degree_level == .
save "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\cip_degree.dta", replace
