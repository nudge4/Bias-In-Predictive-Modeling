global username="`c(username)'"
global data "/Users/${username}/Box Sync/VCCS restricted student data"
global data "D:/Yifeng -- Project Work/ys8mz_sandbox"


** Load the Financial files one by one, and append it to the running merged file
clear
local files : dir "${data}/Raw/FinancialAid" files "*.csv"
foreach file in `files' {
		di "`file'"
		local yr = substr("`file'",2,4)
		preserve
			import delimited using "${data}/Raw/FinancialAid/`file'", clear stringcols(_all)
			duplicates drop
			foreach v in acg ctg ctgplus gearup grsef grsnbef loanef locgov othef othin othoth outinst priloan smart stemef wsot {
				replace `v' = "0" if `v' == "NA"
			}
			gen year = `yr'
			tempfile finaid_file
			save "`finaid_file'", replace
		restore
		append using "`finaid_file'", force
}
drop aidwin athperc calsys credwin exce gender grdcom lastdol level locdomi part race rectype repper repyear stustat tagu tuition vaguap visa vocreh vsp vtg zip // keep "budget" column here!
replace budget = "000000" if budget == "XXXXXX" & pell != "0"
replace budget = "" if budget == "XXXXXX"
replace budget = "" if budget == "000000" & pell == "0.00"
destring acg-wsot, replace
order vccsid year fice aidsum aidfal aidspr budget credfal credspr credsum
egen tot_aids = rowtotal(acg-wsot)
gen aid_avail = (aidfal+aidspr+aidsum > 0)
assert tot_aids == 0 if aid_avail == 0
* drop if tot_aids == 0
sort vccsid year fice
duplicates drop
collapse (max) aidsum aidfal aidspr budget credsum credfal credspr (sum) acg-wsot, by(vccsid year fice)
gen credall = credsum+credfal+credspr
foreach v in "sum" "fal" "spr" {
	replace cred`v' = 1 if credall == 0 & aid`v' == 1
}
order vccsid-credspr credall
foreach v in "sum" "fal" "spr" {
	replace cred`v' = 0 if aid`v' == 0
}
replace credall = credsum + credfal + credspr
foreach v in "sum" "fal" "spr" {
	replace cred`v' = 1 if credall == 0 & aid`v' == 1
}
replace credall = credsum + credfal + credspr
foreach v in "sum" "fal" "spr" {
	gen pct`v' = cred`v'/credall
}
drop aidsum-aidspr 
drop credsum-credall
order vccsid year fice budget pctsum pctfal pctspr
sort vccsid year fice
gen tot_grants = acg + csap + ctg + ctgplus + discaid + gearup + grantin + grsef + grsnbef + hetap + locgov + msdawd + msdtfw + othef + othfed + othin + othoth + outinst + pell + ptap + schoin + seog + smart + tag + tuiwaiv + tviigrants + vgap
gen tot_sub_loans = perkins + staloa
gen tot_unsub_loans = loanef + loanin + plusloa + priloan + staloun + tviiloans
gen tot_others = cwsp + stemef + stemin + wsot
keep vccsid-pctspr pell tot*
rename pell tot_pell
foreach v in pell grants sub_loans unsub_loans others {
	foreach t in sum fal spr {
		gen `v'_`t' = tot_`v' * pct`t'
	}
}
drop pct* tot_*
foreach v in pell grants sub_loans unsub_loans others {
	foreach t in sum fal spr {
		if "`t'" == "sum" {
			rename `v'_`t' `v'3
		}
		else if "`t'" == "fal" {
			rename `v'_`t' `v'4
		}
		else {
			rename `v'_`t' `v'2
		}
	}
}
replace budget = 999999 if budget == .
reshape long pell@ grants@ sub_loans@ unsub_loans@ others@, i(vccsid year fice budget) j(term)
replace year = year+1 if term == 2
replace pell = . if pell == 0 & budget == 999999
sort vccsid year term
collapse (sum) pell grants sub_loans unsub_loans others, by(vccsid year term budget)
replace pell = . if pell == 0 & budget == 999999
replace pell = (pell > 0) if pell != .
collapse (max) pell (sum) grants sub_loans unsub_loans others, by(vccsid year term)
tostring year, replace
tostring term, replace
gen strm = substr(year,1,1) + substr(year,3,2) + term
drop year term
order vccsid strm
destring strm, replace
sort vccsid strm

drop grants-others
drop if strm > 2194
save "${data}/bias_analyses/pell_by_term.dta", replace
