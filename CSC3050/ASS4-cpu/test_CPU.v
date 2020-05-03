/*
test_CPU.v
by Jamie Chen
ver. 2020/5/3
-------------------------------------------
iverilog -o main CPU.v test_CPU.v; vvp main
*/

`timescale 1ns/1ps

// general register
`define gr0  	5'b00000
`define gr1  	5'b00001
`define gr2  	5'b00010
`define gr3 	5'b00011
`define gr4  	5'b00100
`define gr5  	5'b00101
`define gr6  	5'b00110
`define gr7  	5'b00111

/*
#PERIOD  // 0.(F) Updates PC; Fetches instruction
#PERIOD  // 1.(D) Control unit decodes instruciton
#PERIOD  // 2.(E) ALU executes instruction
#PERIOD  // 3.(M) Memory I/O
#PERIOD  // 4.(W) Writeback to register file
*/

module CPU_test;
    parameter CLOCKS = 50;
    parameter PERIOD = 10;
    parameter HALF_PERIOD = PERIOD / 2;
    reg clk;
    always #HALF_PERIOD clk = ~clk;

    alu uut(
        .clock(clk)
    );

initial begin
// Data Memory
uut.dataMemory[0] = 32'h0000_00ab; // address 0x00
uut.dataMemory[1] = 32'h0000_3c00; // address 0x04

// Instruction Memory
uut.instrMemory[0] = {6'b100011, `gr0, `gr1, 16'h0000};   // lw gr1, gr0(0)    (gr1 <= memory[0x00])
uut.instrMemory[1] = {6'b100011, `gr0, `gr2, 16'h0004};   // lw gr2, gr0(4)    (gr2 <= memory[0x04])
uut.instrMemory[2] = {6'b101011, `gr0, `gr0, 16'h0008};   // sw gr0, gr0(8)    (data[0x08] <= 0)
uut.instrMemory[3] = {6'b001000, `gr0, `gr3, 16'h0001};   // addi gr3, gr0, 1  (gr3 <= 0 + 1)
uut.instrMemory[4] = {6'b001001, `gr0, `gr3, 16'h0002};   // addiu gr3, gr0, 2 (gr3 <= 0 + 2)
uut.instrMemory[5] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100000};  // add gr3, gr1, gr2 (gr3 <= gr1 + gr2)
uut.instrMemory[6] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100010};  // sub gr3, gr1, gr2 (gr3 <= gr1 - gr2)
uut.instrMemory[7] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100001};  // addu gr3, gr1, gr2 (gr3 <= gr1 + gr2)
uut.instrMemory[8] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100011};  // subu gr3, gr1, gr2 (gr3 <= gr1 - gr2)
uut.instrMemory[9] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100100};  // and gr3, gr1, gr2 (gr3 <= gr1 & gr2)
uut.instrMemory[10] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100101};  // or gr3, gr1, gr2 (gr3 <= gr1 | gr2)
uut.instrMemory[11] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100111};  // nor gr3, gr1, gr2 (gr3 <= gr1 ~| gr2)
uut.instrMemory[12] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100110};  // xor gr3, gr1, gr2 (gr3 <= gr1 ^ gr2)
uut.instrMemory[13] = {6'b001100, `gr0, `gr3, 16'h1111};   // andi gr3, gr0, 0x1111  (gr3 <= 0 & 0x1111)
uut.instrMemory[14] = {6'b001101, `gr0, `gr3, 16'h1111};   // ori gr3, gr0, 0x1111 (gr3 <= 0 | 0x1111)
uut.instrMemory[15] = {6'b000000, `gr0, `gr1, `gr3, 5'b00001, 6'b000000};  // sll gr3, gr1, 1 (gr3 <= gr1 << 1)
uut.instrMemory[16] = {6'b000000, `gr0, `gr1, `gr3, 5'b00001, 6'b000010};  // srl gr3, gr1, 1 (gr3 <= gr1 >> 1)
uut.instrMemory[17] = {6'b000000, `gr0, `gr1, `gr3, 5'b00001, 6'b000011};  // sra gr3, gr1, 1 (gr3 <= gr1 >>> 1)
uut.instrMemory[18] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b000100};  // sllv gr3, gr2, gr1 (gr3 <= gr2 << gr1)
uut.instrMemory[19] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b000110};  // srlv gr3, gr2, gr1 (gr3 <= gr2 >> gr1)
uut.instrMemory[20] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b000111};  // srav gr3, gr2, gr1 (gr3 <= gr2 >>> gr1)
uut.instrMemory[21] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b101010};  // slt gr3, gr1, gr2 (gr3 <= gr1 < gr2)
uut.instrMemory[22] = {6'b000100, `gr1, `gr2, 16'hffff};  // beq gr1, gr2, -1 (branch -1 if gr1 - gr2 == 0)
uut.instrMemory[26] = {6'b000101, `gr1, `gr2, 16'h0003};  // bne gr0, gr0, 3 (branch 3 if gr1 - gr2 != 0)
uut.instrMemory[30] = {6'b000000, `gr1, `gr2, `gr3, 5'b00000, 6'b100000};  // add gr3, gr1, gr2 (gr3 <= gr1 + gr2)
uut.instrMemory[31] = {6'b000010, 26'h000008c};  // j 0x8c (jump to 0x8c unconditionally)
uut.instrMemory[35] = {6'b000011, 26'h000009c};  // jal 0x9c (jump to 0x9c and link)
uut.instrMemory[39] = {6'b000000, `gr0, 15'h0000, 6'h8};  // jr gr0 (jump to gr0 unconditionally)

// Initialization
uut.PC = 0;
uut.gr[0] = 0;
clk = 1;

    $display("PCF     : instrD :  srcAE :  srcBE :ALUOutE :writeDataM:readDataM: resultW:  gr1   :  gr2   :  gr3   ");
    $monitor("%h:%h:%h:%h:%h: %h :%h :%h:%h:%h:%h",
    uut.PC, uut.instr, uut.srcA, uut.srcB, uut.ALUResult[0], uut.writeData[1], uut.readData[0],uut.result, uut.gr[1], uut.gr[2], uut.gr[3]);

    # (CLOCKS * PERIOD)

    $finish;
    end

endmodule