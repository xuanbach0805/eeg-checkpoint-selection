// ============================================================================
//  tb_eeg_lin_clf.v — doi chieu RTL voi vector vang sinh tu Python.
//  Vector vang chinh la argmax cua duong so nguyen trong `gen_rtl_data.py`,
//  nen bat ky sai lech nao deu la LOI RTL, khong phai sai so luong tu hoa.
// ============================================================================
`default_nettype none
`timescale 1ns/1ps

module tb_eeg_lin_clf;

`include "params.vh"

    localparam integer DIM   = `DIM;
    localparam integer WBITS = `WBITS;
    localparam integer XBITS = `XBITS;
    localparam integer ACCW  = `ACCW;
    localparam integer ADDRW = `ADDRW;
    localparam integer NSTIM = `NSTIM;

    reg clk = 1'b0, rst_n = 1'b0, x_we = 1'b0, start = 1'b0;
    reg [ADDRW-1:0]        x_addr = 0;
    reg signed [XBITS-1:0] x_din  = 0;
    wire done;
    wire [1:0] class_out;

    always #5 clk = ~clk;                     // 100 MHz trong mo phong

    eeg_lin_clf #(.DIM(DIM), .WBITS(WBITS), .XBITS(XBITS), .ACCW(ACCW), .ADDRW(ADDRW),
                  .W0FILE("w_c0.hex"), .W1FILE("w_c1.hex"),
                  .W2FILE("w_c2.hex"), .BFILE("bias.hex"))
    dut (.clk(clk), .rst_n(rst_n), .x_we(x_we), .x_addr(x_addr), .x_din(x_din),
         .start(start), .done(done), .class_out(class_out));

    reg signed [XBITS-1:0] stim   [0:NSTIM*DIM-1];
    reg [1:0]              golden [0:NSTIM-1];

    integer i, k, errors, cycles, total_cycles;

    initial begin
        $readmemh("stimulus.hex", stim);
        $readmemh("golden.txt",   golden);    // moi dong la 0/1/2

        errors = 0; total_cycles = 0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        for (i = 0; i < NSTIM; i = i + 1) begin
            for (k = 0; k < DIM; k = k + 1) begin   // nap vector dac trung
                @(negedge clk);
                x_we = 1'b1; x_addr = k[ADDRW-1:0]; x_din = stim[i*DIM + k];
            end
            @(negedge clk); x_we = 1'b0;

            @(negedge clk); start = 1'b1;           // khoi dong
            @(negedge clk); start = 1'b0;

            cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles = cycles + 1;
                if (cycles > DIM + 100) begin
                    $display("LOI: cua so %0d khong bao done sau %0d chu ky", i, cycles);
                    $finish;
                end
            end
            total_cycles = total_cycles + cycles;

            if (class_out !== golden[i]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("SAI  cua so %0d: RTL=%0d vang=%0d", i, class_out, golden[i]);
            end
            @(negedge clk);
        end

        $display("");
        $display("================ KET QUA ================");
        $display("DIM=%0d WBITS=%0d XBITS=%0d ACCW=%0d", DIM, WBITS, XBITS, ACCW);
        $display("So cua so kiem tra : %0d", NSTIM);
        $display("So sai             : %0d", errors);
        $display("Do tre trung binh  : %0d chu ky/cua so", total_cycles / NSTIM);
        if (errors == 0) $display("KET LUAN: PASS — RTL khop 100%% vector vang.");
        else             $display("KET LUAN: FAIL");
        $display("=========================================");
        $finish;
    end

endmodule

`default_nettype wire
