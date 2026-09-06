# ============================================================================
#  synth_all.tcl — Tong hop TAT CA cau hinh bo phan loai, chi bang PHAN MEM.
#  Khong can board, khong can nap bitstream.
#
#  Chay:   quartus_sh -t synth_all.tcl
#  hoac ep thiet bi:   quartus_sh -t synth_all.tcl EP2C35F672C6 "Cyclone II"
#
#  Ket qua: ket_qua_tong_hop.csv     Nhat ky loi: build/loi_<buoc>.txt
# ============================================================================

set ROOT   [file normalize [file dirname [info script]]]
set DATA   [file normalize "$ROOT/../rtl_data"]
set CONFIGS {C1_1ch_w2 C2_4ch_w4 C3_8ch_w6 C4_32ch_w6 C5_62ch_w8}

# ---------------------------------------------------------------------------
#  THU MUC BUILD PHAI NAM NGOAI GOOGLE DRIVE / ONEDRIVE / DROPBOX.
#  Quartus tao roi xoa thu muc `incremental_db` lien tuc trong luc chay; cac
#  dich vu dong bo khoa thu muc moi tao -> Error (138001) "Cannot write to
#  directory". Vi vay build o o dia local, chi ghi CSV ket qua ve $ROOT.
# ---------------------------------------------------------------------------
proc writable {dir} {
    if {[catch {file mkdir $dir}]} { return 0 }
    set probe [file join $dir _ghi_thu.tmp]
    if {[catch {set fh [open $probe w]}]} { return 0 }
    catch {puts $fh "ok"; close $fh}
    if {[catch {file mkdir [file join $dir _thu_muc_con]}]} { return 0 }
    catch {file delete -force $probe [file join $dir _thu_muc_con]}
    return 1
}
set BUILD ""
set CANDS {}
if {[info exists env(ED_BUILD)]} { lappend CANDS $env(ED_BUILD) }
lappend CANDS "C:/qbuild_ed"
if {[info exists env(TEMP)]}      { lappend CANDS "[string map {\\ /} $env(TEMP)]/qbuild_ed" }
if {[info exists env(LOCALAPPDATA)]} { lappend CANDS "[string map {\\ /} $env(LOCALAPPDATA)]/qbuild_ed" }
lappend CANDS "$ROOT/build"
foreach c $CANDS {
    if {[writable $c]} { set BUILD $c; break }
    puts "  (khong ghi duoc vao $c)"
}
if {$BUILD eq ""} { puts "DUNG LAI: khong tim duoc thu muc build ghi duoc."; exit 1 }

# ho thiet bi uu tien, theo thu tu; chi dung nhung ho THUC SU da cai tren may
set FAM_PREF {"Cyclone II" "Cyclone IV E" "Cyclone III" "Cyclone IV GX" "Cyclone" \
              "MAX 10" "MAX II" "Stratix IV" "Arria II GX"}
set DEV_PREF {EP2C35F672C6 EP2C70F896C6 EP4CE115F29C7 EP4CE22F17C6 EP3C16F484C6}

set DEVICE ""
set FAMILY ""
set _a {}
if {[info exists quartus(args)]} { set _a $quartus(args) }
if {[llength $_a] >= 2} {
    set DEVICE [lindex $_a 0]
    set FAMILY [lindex $_a 1]
}

puts "============================================================"
puts " Du lieu : $DATA"
puts " Build   : $BUILD   (ngoai Google Drive — bat buoc)"
puts "============================================================"


proc read_defines {file} {
    set out [dict create]
    set fh [open $file r]
    foreach line [split [read $fh] "\n"] {
        if {[regexp {^\s*`define\s+(\w+)\s+(\S+)} $line -> k v]} { dict set out $k $v }
    }
    close $fh
    return $out
}

proc write_project {dir cfg device family P} {
    # Quartus can CA .qpf lan .qsf
    set qpf [open "$dir/$cfg.qpf" w]
    puts $qpf "QUARTUS_VERSION = \"13.0\""
    puts $qpf "PROJECT_REVISION = \"$cfg\""
    close $qpf

    set qsf [open "$dir/$cfg.qsf" w]
    puts $qsf "set_global_assignment -name FAMILY \"$family\""
    puts $qsf "set_global_assignment -name DEVICE $device"
    puts $qsf "set_global_assignment -name TOP_LEVEL_ENTITY eeg_lin_clf"
    puts $qsf "set_global_assignment -name ORIGINAL_QUARTUS_VERSION 13.0"
    puts $qsf "set_global_assignment -name VERILOG_FILE eeg_lin_clf.v"
    puts $qsf "set_global_assignment -name SDC_FILE eeg_lin_clf.sdc"
    puts $qsf "set_global_assignment -name SEARCH_PATH \".\""
    foreach k {DIM WBITS XBITS ACCW ADDRW} {
        puts $qsf "set_parameter -name $k [dict get $P $k]"
    }
    close $qsf
}

proc grab {file pattern} {
    if {![file exists $file]} { return "n/a" }
    set fh [open $file r]; set txt [read $fh]; close $fh
    if {[regexp $pattern $txt -> m]} {
        # Quartus in so kieu "1,600" -> bo dau phay, neu khong CSV se lech cot
        return [string map {, {}} [string trim $m]]
    }
    return "n/a"
}

# --- IN RA LOI THAT: Quartus de loi o CUOI, dau ra chi la banner ---
proc show_error {msg logfile} {
    set errs {}
    foreach line [split $msg "\n"] {
        if {[regexp {^\s*(Error|Critical Warning)} $line]} { lappend errs [string trim $line] }
    }
    if {[llength $errs] == 0} {
        # khong co dong "Error" -> in 20 dong CUOI
        set L [split [string trim $msg] "\n"]
        set n [llength $L]
        set errs [lrange $L [expr {$n > 20 ? $n-20 : 0}] end]
    }
    if {[llength $errs] > 12} { set errs [lrange $errs 0 11] }
    foreach e $errs { puts "      $e" }
    set fh [open $logfile w]; puts $fh $msg; close $fh
    puts "      (nhat ky day du: $logfile)"
}


# ---------------- HOI QUARTUS xem may nay cai nhung thiet bi gi ----------------
if {$DEVICE eq ""} {
    puts "\n--- Cac ho thiet bi da cai tren may nay ---"
    set fams {}
    if {[catch {load_package device} e]} { puts "  (khong nap duoc goi device: $e)" }
    if {[catch {set fams [get_family_list]} e]} { puts "  (khong liet ke duoc: $e)"; set fams {} }
    if {[llength $fams] == 0} {
        puts "  Quartus khong tra ve ho thiet bi nao."
    } else {
        foreach f [lsort $fams] { puts "  - $f" }
    }

    # 1) uu tien: mot trong cac thiet bi ta muon, NEU no co trong danh sach cai dat
    foreach f $fams {
        set parts {}
        catch {set parts [get_part_list -family $f]}
        foreach want $DEV_PREF {
            if {[lsearch -exact $parts $want] >= 0} {
                set DEVICE $want; set FAMILY $f; break
            }
        }
        if {$DEVICE ne ""} break
    }
    # 2) neu khong, lay thiet bi LON NHAT trong ho uu tien dau tien co mat
    if {$DEVICE eq ""} {
        foreach pf $FAM_PREF {
            if {[lsearch -exact $fams $pf] < 0} { continue }
            set parts {}
            catch {set parts [get_part_list -family $pf]}
            if {[llength $parts] == 0} { continue }
            set best ""; set bestn -1
            foreach p $parts {
                set n 0
                catch {set n [get_part_info -lcell_count $p]}
                if {$n > $bestn} { set bestn $n; set best $p }
            }
            if {$best eq ""} { set best [lindex $parts 0] }
            set DEVICE $best; set FAMILY $pf
            break
        }
    }
    # 3) cuoi cung: ho bat ky
    if {$DEVICE eq "" && [llength $fams] > 0} {
        set f [lindex $fams 0]
        set parts {}
        catch {set parts [get_part_list -family $f]}
        if {[llength $parts] > 0} { set DEVICE [lindex $parts 0]; set FAMILY $f }
    }
}

if {$DEVICE eq ""} {
    puts "\nDUNG LAI: khong xac dinh duoc thiet bi nao. Gui toi toan bo thong bao tren."
    exit 1
}
puts "\n>>> Dung thiet bi: $DEVICE ($FAMILY)\n"


# ---------------- tong hop tung cau hinh ----------------
file mkdir $BUILD
set here [pwd]
set rows {}
foreach cfg $CONFIGS {
    set src "$DATA/$cfg"
    if {![file isdirectory $src]} { puts "BO QUA $cfg (khong thay $src)"; continue }
    set d "$BUILD/$cfg"
    file delete -force $d
    file mkdir $d
    foreach f [glob -nocomplain "$src/*.hex"] { file copy -force $f $d }
    file copy -force "$ROOT/eeg_lin_clf.v"   $d
    file copy -force "$ROOT/eeg_lin_clf.sdc" $d
    file copy -force "$ROOT/fmax.tcl"        $d
    set P [read_defines "$src/params.vh"]
    write_project $d $cfg $DEVICE $FAMILY $P

    puts "--- $cfg (DIM=[dict get $P DIM] WBITS=[dict get $P WBITS] ACCW=[dict get $P ACCW]) ---"
    cd $d
    set ok 1
    # Chi quartus_map va quartus_fit nhan --read_settings_files/--write_settings_files.
    # quartus_sta KHONG nhan -> Error (23024) "Unknown long option".
    set STEPS [list \
        [list quartus_map [list --read_settings_files=on --write_settings_files=off $cfg -c $cfg]] \
        [list quartus_fit [list --read_settings_files=on --write_settings_files=off $cfg -c $cfg]] \
        [list quartus_sta [list $cfg -c $cfg]] ]
    foreach sp $STEPS {
        set step [lindex $sp 0]
        set argv [lindex $sp 1]
        if {[catch {eval exec $step $argv} msg]} {
            puts "  LOI o $step:"
            cd $here
            show_error $msg "$BUILD/loi_${cfg}_${step}.txt"
            cd $d
            set ok 0
            break
        }
        puts "  $step OK"
    }
    # fmax.tcl chay rieng: quartus_sta -t <script> <project> <revision>
    if {$ok} {
        if {[catch {exec quartus_sta -t fmax.tcl $cfg $cfg} m2]} {
            cd $here
            puts "  (canh bao: fmax.tcl khong chay duoc, se lay Fmax tu $cfg.sta.rpt)"
            show_error $m2 "$BUILD/loi_${cfg}_fmax.txt"
            cd $d
        }
    }
    cd $here
    if {!$ok} { continue }

    set fitsum "$d/$cfg.fit.summary"
    set le   [grab $fitsum {Total logic elements\s*:\s*([\d,]+)}]
    if {$le eq "n/a"} { set le [grab $fitsum {Total combinational functions\s*:\s*([\d,]+)}] }
    set alm  [grab $fitsum {Logic utilization \(in ALMs\)\s*:\s*([\d,]+)}]
    set ff   [grab $fitsum {Total registers\s*:\s*([\d,]+)}]
    set mem  [grab $fitsum {Total memory bits\s*:\s*([\d,]+)}]
    set mult [grab $fitsum {Embedded Multiplier 9-bit elements\s*:\s*([\d,]+)}]
    if {$mult eq "n/a"} { set mult [grab $fitsum {Total DSP Blocks\s*:\s*([\d,]+)}] }
    # lay RIENG con so, khong kem don vi -> cot CSV la so thuc, tien tinh toan
    set fmax [grab "$d/fmax.txt" {;\s*([\d.]+)\s*MHz\s*;}]
    if {$fmax eq "n/a"} { set fmax [grab "$d/$cfg.sta.rpt" {;\s*([\d.]+)\s*MHz\s*;\s*[\d.]+\s*MHz\s*;\s*clk}] }

    lappend rows [list $cfg [dict get $P DIM] [dict get $P WBITS] [dict get $P XBITS] \
                       [dict get $P ACCW] $le $alm $ff $mem $mult $fmax \
                       [expr {[dict get $P DIM] + 4}]]
    puts "  LE=$le ALM=$alm FF=$ff mem=$mem nhan=$mult Fmax=$fmax MHz\n"
}

if {[llength $rows] == 0} {
    puts "KHONG cau hinh nao tong hop duoc. Gui toi thong bao loi o tren"
    puts "va cac tep build/loi_*.txt."
    exit 1
}

set csv [open "$ROOT/ket_qua_tong_hop.csv" w]
puts $csv "device,family,config,DIM,WBITS,XBITS,ACCW,logic_elements,ALMs,registers,memory_bits,multipliers,Fmax_MHz,latency_cycles"
foreach r $rows { puts $csv "$DEVICE,\"$FAMILY\",[join $r ","]" }
close $csv

puts "============================================================"
puts " XONG. Thiet bi: $DEVICE ($FAMILY)"
puts " Ket qua: $ROOT/ket_qua_tong_hop.csv"
puts "============================================================"
foreach r $rows {
    puts [format "  %-12s LE=%-8s FF=%-6s nhan=%-4s Fmax=%s MHz" \
              [lindex $r 0] [lindex $r 5] [lindex $r 7] [lindex $r 9] [lindex $r 10]]
}
