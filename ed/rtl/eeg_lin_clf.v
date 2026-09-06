// ============================================================================
//  eeg_lin_clf.v — Bo phan loai cam xuc EEG tuyen tinh, so nguyen hoan toan.
//
//  acc[c] = SUM_{d=0..DIM-1} ( x[d] * w[c][d] ) + bias[c]
//  class  = argmax_c acc[c]            (hoa -> chon chi so NHO NHAT, giong numpy.argmax)
//
//  Trong so dung MOT he so ti le chung cho ca 3 lop (`tensor_max`), nho vay argmax
//  thuc hien truc tiep tren thanh ghi tich luy so nguyen — khong can nhan de-quantize.
//  Do la ly do thiet ke nay dung tensor_max chu khong phai row_mse.
//
//  Cac tep .hex do `gen_rtl_data.py` sinh ra; be rong ACCW da duoc chon theo truong hop
//  xau nhat tuyet doi va kiem chung khong tran tren toan bo tap test.
//
//  3 nhan song song (mot cho moi lop), tuan tu theo chieu d  ->  do tre = DIM + 4 chu ky.
// ============================================================================
`default_nettype none

module eeg_lin_clf #(
    parameter integer DIM   = 20,      // so chieu dac trung  (so kenh x 5 bang tan)
    parameter integer WBITS = 4,       // be rong trong so
    parameter integer XBITS = 10,      // be rong dac trung dau vao
    parameter integer ACCW  = 18,      // be rong thanh ghi tich luy
    parameter integer ADDRW = 5,       // ceil(log2(DIM))
    parameter W0FILE = "w_c0.hex",
    parameter W1FILE = "w_c1.hex",
    parameter W2FILE = "w_c2.hex",
    parameter BFILE  = "bias.hex"
)(
    input  wire                    clk,
    input  wire                    rst_n,
    // ghi vector dac trung tu host
    input  wire                    x_we,
    input  wire [ADDRW-1:0]        x_addr,
    input  wire signed [XBITS-1:0] x_din,
    // dieu khien
    input  wire                    start,
    output reg                     done,
    output reg  [1:0]              class_out   // 0=negative 1=neutral 2=positive
);

    // ---------------- bo nho ----------------
    reg signed [XBITS-1:0] xbuf [0:DIM-1];
    reg signed [WBITS-1:0] w0   [0:DIM-1];
    reg signed [WBITS-1:0] w1   [0:DIM-1];
    reg signed [WBITS-1:0] w2   [0:DIM-1];
    reg signed [ACCW-1:0]  bias [0:2];

    initial begin
        $readmemh(W0FILE, w0);
        $readmemh(W1FILE, w1);
        $readmemh(W2FILE, w2);
        $readmemh(BFILE,  bias);
    end

    always @(posedge clk) begin
        if (x_we) xbuf[x_addr] <= x_din;
    end

    // ---------------- may trang thai ----------------
    localparam [1:0] S_IDLE = 2'd0, S_MAC = 2'd1, S_ARGMAX = 2'd2, S_DONE = 2'd3;

    reg [1:0]            state;
    reg [ADDRW:0]        d;                 // rong hon 1 bit de dem den DIM
    reg signed [ACCW-1:0] acc0, acc1, acc2;

    wire signed [XBITS-1:0] xv = xbuf[d[ADDRW-1:0]];
    wire signed [XBITS+WBITS-1:0] p0 = xv * w0[d[ADDRW-1:0]];
    wire signed [XBITS+WBITS-1:0] p1 = xv * w1[d[ADDRW-1:0]];
    wire signed [XBITS+WBITS-1:0] p2 = xv * w2[d[ADDRW-1:0]];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 1'b0; class_out <= 2'd0;
            d <= {(ADDRW+1){1'b0}}; acc0 <= 0; acc1 <= 0; acc2 <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        acc0  <= bias[0];
                        acc1  <= bias[1];
                        acc2  <= bias[2];
                        d     <= {(ADDRW+1){1'b0}};
                        state <= S_MAC;
                    end
                end
                S_MAC: begin
                    acc0 <= acc0 + $signed({{(ACCW-XBITS-WBITS){p0[XBITS+WBITS-1]}}, p0});
                    acc1 <= acc1 + $signed({{(ACCW-XBITS-WBITS){p1[XBITS+WBITS-1]}}, p1});
                    acc2 <= acc2 + $signed({{(ACCW-XBITS-WBITS){p2[XBITS+WBITS-1]}}, p2});
                    if (d == DIM - 1) state <= S_ARGMAX;
                    d <= d + 1'b1;
                end
                S_ARGMAX: begin
                    // hoa -> chi so nho nhat thang (dung quy uoc cua numpy.argmax)
                    if ((acc0 >= acc1) && (acc0 >= acc2))      class_out <= 2'd0;
                    else if (acc1 >= acc2)                     class_out <= 2'd1;
                    else                                       class_out <= 2'd2;
                    state <= S_DONE;
                end
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
