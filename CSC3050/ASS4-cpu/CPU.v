/* 
CPU.v
by Jamie Chen
ver. 2020/5/3
-------------------------------------------
iverilog -o main CPU.v test_CPU.v; vvp main
*/

`timescale 1ns/1ps

`define rt 0
`define rs 1
`define rd 1
`define imm 2
`define signImm 3
`define shamt 4

`define add 0
`define sub 1
/*
`define mult 2
`define multu 3
`define div 4
`define divu 5
*/
`define AND 2
`define NOR 3
`define OR 4
`define XOR 5
`define slt 6
`define sltu 7
`define sll 8
`define srl 9
`define sra 10

module alu(
	input wire clock
);

// REGISTER I/O
reg[31 : 0] gr[7:0];
reg[31 : 0] result;
reg[4  : 0] writeReg[2:0];

// MEMORY I/O
reg[31 : 0] dataMemory[10000:0];
reg[31 : 0] instrMemory[1000:0];
reg[31 : 0] readData[1:0];
reg[31 : 0] writeData[1:0];

// INSTRUCTOIN SPLITING
reg[31: 0] instr;
reg[5 : 0] opcode, funct; // operation and function codes
reg[5 : 0] rs[1:0], rt[1:0], rd[1:0]; // register number, NOT NEEDED IN THIS PROJECT
reg[5 : 0] shamt[1:0]; // shift amount
reg[15: 0] imm[1:0]; // immediate value
reg[31: 0] signImm[1:0]; // sign extended imm
reg[31: 0] PCTarg[2:0]; // jump target (zero-extended last 26 bits)

// CONTROL UNITS
reg branchCtrl;
reg regWrite[3 : 0];
reg memToReg[3 : 0];
reg jumpLink[3 : 0];
reg jumpReg [2 : 0];
reg jumpTarg[2 : 0];
reg memWrite[2 : 0];
reg branchEq[2 : 0];
reg branchNE[2 : 0];
integer ALUctrl [1 : 0];
reg 	srcActrl[1 : 0];
integer srcBctrl[1 : 0];
reg 	regDst  [1 : 0];

// MAIN ALU
reg[31 : 0] srcA;
reg[31 : 0] srcB;
reg[31 : 0] ALUResult[2 : 0];
reg zeroFlag[1 : 0];

// PROGRAM COUNTER
reg[31 : 0] PC;
reg[31 : 0] PCPlus4[4 : 0];
reg[31 : 0] PCBranch[1 : 0];
reg[31 : 0] PCReg[1 : 0];

// combinational logic
always @(*) begin
	// PC Management
	PCPlus4[0] = PC + 4;
	PCBranch[0] = PCPlus4[2] + (signImm[1] << 2);
	branchCtrl = (zeroFlag[1] & branchEq[2]) | (!zeroFlag[1] & branchNE[2]);

	// Instruction Splitting & Sign Extension
	opcode = instr[31 : 26];
	funct  = instr[05 : 00];
	rs[0]  = instr[25 : 21];
	rt[0]  = instr[20 : 16];
	rd[0]  = instr[15 : 11];
	shamt[0]  = instr[10 : 06];
	imm[0]    = instr[15 : 00];
	PCTarg[0] = instr[25 : 00];
	signImm[0] = $signed(imm[0]);

	// Control Unit
	jumpLink[0] = 0; // default: no jal
	jumpTarg[0] = 0; // default: no jump
	jumpReg [0] = 0; // default: no jump
	branchEq[0] = 0; // default: no branch
	branchNE[0] = 0; // default: no branch
	memWrite[0] = 0; // default: disable memory write
	regWrite[0] = 1; // default: enable reg write
	memToReg[0] = 0; // default: if write to reg, result = ALUResult

	case(opcode)
		0:	// R-type
		begin
		regDst[0] = `rd;
		case(funct)
			0:  //sll
			begin
			ALUctrl[0] = `sll;
			srcActrl[0] = `rt;
			srcBctrl[0] = `shamt;
			end

			2:  //srl
			begin
			ALUctrl[0] = `srl;
			srcActrl[0] = `rt;
			srcBctrl[0] = `shamt;
			end

			3:  // sra
			begin
			ALUctrl[0] = `sra;
			srcActrl[0] = `rt;
			srcBctrl[0] = `shamt;
			end

			4:  // sllv
			begin
			ALUctrl[0] = `sll;
			srcActrl[0] = `rt;
			srcBctrl[0] = `rs;
			end
			
			6:  // srlv
			begin
			ALUctrl[0] = `srl;
			srcActrl[0] = `rt;
			srcBctrl[0] = `rs;
			end

			7:  // srav
			begin
			ALUctrl[0] = `sra;
			srcActrl[0] = `rt;
			srcBctrl[0] = `rs;
			end

			8:	// jr
			begin
			ALUctrl[0] = 1'bx;
			regWrite[0] = 0;
			jumpReg[0] = 1;
			end

			8'h20:  // add
			begin
			ALUctrl[0] = `add;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

			8'h21:	// addu
			begin
			ALUctrl[0] = `add;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

			8'h22:  // sub
			begin
			ALUctrl[0] = `sub;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

			8'h23:  // subu
			begin
			ALUctrl[0] = `sub;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

			8'h24:  // and
			begin
			ALUctrl[0] = `AND;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

			8'h25:  // or
			begin
			ALUctrl[0] = `OR;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

			8'h26:  // xor
			begin
			ALUctrl[0] = `XOR;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

			8'h27:  // nor
			begin
			ALUctrl[0] = `NOR;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

			8'h2a:  // slt
			begin
			ALUctrl[0] = `slt;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end
			
			8'h2b:  // sltu
			begin
			ALUctrl[0] = `sltu;
			srcActrl[0] = `rs;
			srcBctrl[0] = `rt;
			end

		endcase

		end

		2:  // j
		begin
		ALUctrl[0] = 1'bx;
		srcActrl[0] = 1'bx;
		srcBctrl[0] = 1'bx;
		regWrite[0] = 0;
		jumpTarg[0] = 1;
		end

		3:  // jal
		begin
		ALUctrl[0] = 1'bx;
		srcActrl[0] = 1'bx;
		srcBctrl[0] = 1'bx;
		jumpTarg[0] = 1;
		jumpLink[0] = 1;
		end

		4:  // beq
		begin
		ALUctrl[0] = `sub;
		srcActrl[0] = `rs;
		srcBctrl[0] = `rt;
		regWrite[0] = 0;
		branchEq[0] = 1;
		end

		5:  // bne
		begin
		ALUctrl[0] = `sub;
		srcActrl[0] = `rs;
		srcBctrl[0] = `rt;
		regWrite[0] = 0;
		branchNE[0] = 1;
		end
		
		8:  // addi
		begin
		ALUctrl[0] = `add;
		srcActrl[0] = `rs;
		srcBctrl[0] = `signImm;
		regDst[0] = `rt;
		end

		9:  // addiu
		begin
		ALUctrl[0] = `add;
		srcActrl[0] = `rs;
		srcBctrl[0] = `signImm;
		regDst[0] = `rt;
		end

		8'h0a:  // slti
		begin
		ALUctrl[0] = `slt;
		srcActrl[0] = `rs;
		srcBctrl[0] = `signImm;
		regDst[0] = `rt;
		end

		8'h0b:  // sltiu
		begin
		ALUctrl[0] = `sltu;
		srcActrl[0] = `rs;
		srcBctrl[0] = `signImm;
		regDst[0] = `rt;
		end
		
		8'h0c:  // andi
		begin
		ALUctrl[0] = `AND;
		srcActrl[0] = `rs;
		srcBctrl[0] = `imm;
		regDst[0] = `rt;
		end

		8'h0d:  // ori
		begin
		ALUctrl[0] = `OR;
		srcActrl[0] = `rs;
		srcBctrl[0] = `imm;
		regDst[0] = `rt;
		end

		8'h0e:  // xori
		begin
		ALUctrl[0] = `XOR;
		srcActrl[0] = `rs;
		srcBctrl[0] = `imm;
		regDst[0] = `rt;
		end

		8'h23:  // lw
		begin
		ALUctrl[0] = `add;
		srcActrl[0] = `rs;
		srcBctrl[0] = `signImm;
		regDst[0] = `rt;
		memToReg[0] = 1; // source = memory
		end

		8'h2b:  // sw
		begin
		ALUctrl[0] = `add;
		srcActrl[0] = `rs;
		srcBctrl[0] = `signImm;
		regWrite[0] = 0; // disables register write
		memWrite[0] = 1; // enables memory write
		end

	endcase

	case (srcActrl[1])
		`rs: srcA = gr[rs[1]];
		`rt: srcA = gr[rt[1]];
		default: srcA = 32'hx;
	endcase
	case (srcBctrl[1])
		`rs:  	 srcB = gr[rs[1]];
		`rt:   	 srcB = gr[rt[1]];
		`imm: 	 srcB = imm[1];
		`shamt:  srcB = shamt[1];
		`signImm:srcB = signImm[1];
		default: srcB = 32'hx;
	endcase

	// main ALU
	case (ALUctrl[1])
		`add: ALUResult[0] = srcA + srcB;
		`sub: ALUResult[0] = srcA - srcB;
		`AND: ALUResult[0] = srcA & srcB;
		`NOR: ALUResult[0] = srcA ~| srcB;
		`OR:  ALUResult[0] = srcA | srcB;
		`XOR: ALUResult[0] = srcA ^ srcB;
		`slt: ALUResult[0] = $signed(srcA) < $signed(srcB);
		`sltu:ALUResult[0] = srcA < (srcB & 32'h1f); // shamt reduced mod 32
		`sll: ALUResult[0] = srcA << (srcB & 32'h1f); // shamt reduced mod 32
		`srl: ALUResult[0] = srcA >> (srcB & 32'h1f); // shamt reduced mod 32
		`sra: ALUResult[0] = srcA >>> (srcB & 32'h1f); // shamt reduced mod 32
		default: ALUResult[0] = 32'hx;
	endcase
	zeroFlag[0] = (ALUResult[0] == 0);

	PCReg	 [0] = gr[rs[1]];
	writeReg [0] = jumpLink[1]? 5'b00011 : (regDst[1]? rd[1] : rt[1]);
	writeData[0] = gr[rt[1]];
	readData [0] = dataMemory[ALUResult[1]/4]; // memory read

	result = jumpLink[3]? PCPlus4[4] : (memToReg[3]? readData[1] : ALUResult[2]);
	if (regWrite[3]) gr[writeReg[2]] = result; 	// writeback to register file
	if (memWrite[2]) dataMemory[ALUResult[1]/4] = writeData[1]; // memory write

end

// sequential logic
always @(posedge clock) begin

	if (branchCtrl) PC <= PCBranch[1]; 
	else if (jumpReg[2]) PC <= PCReg[1];
	else if (jumpTarg[2]) PC <= PCTarg[2];
	else PC <= PCPlus4[0];

	instr <= instrMemory[PC / 4];

	PCBranch[1] <= PCBranch[0];
	imm[1] 	    <= imm[0];
	signImm[1]  <= signImm[0];
	rs[1]       <= rs[0];
	rt[1]       <= rt[0];
	rd[1]       <= rd[0];
	shamt[1]    <= shamt[0];
	PCReg[1] 	<= PCReg[0];
	PCTarg[1]   <= PCTarg[0];
	PCTarg[2]   <= PCTarg[1];

	PCPlus4 [1] <= PCPlus4 [0];
	PCPlus4 [2] <= PCPlus4 [1];
	PCPlus4 [3] <= PCPlus4 [2];
	PCPlus4 [4] <= PCPlus4 [3];

	regWrite[1] <= regWrite[0];
	regWrite[2] <= regWrite[1];
	regWrite[3] <= regWrite[2];

	memToReg[1] <= memToReg[0];
	memToReg[2] <= memToReg[1];
	memToReg[3] <= memToReg[2];

	jumpLink[1] <= jumpLink[0];
	jumpLink[2] <= jumpLink[1];
	jumpLink[3] <= jumpLink[2];

	jumpReg[1] <= jumpReg[0];
	jumpReg[2] <= jumpReg[1];

	jumpTarg[1] <= jumpTarg[0];
	jumpTarg[2] <= jumpTarg[1];

	memWrite[1] <= memWrite[0];
	memWrite[2] <= memWrite[1];

	writeReg[1] <= writeReg[0];
	writeReg[2] <= writeReg[1];

	branchEq[1] <= branchEq[0];
	branchEq[2] <= branchEq[1];

	branchNE[1] <= branchNE[0];
	branchNE[2] <= branchNE[1];

	ALUResult[1]<= ALUResult[0];
	ALUResult[2]<= ALUResult[1];

	ALUctrl [1] <= ALUctrl [0];

	srcActrl[1] <= srcActrl[0];

	srcBctrl[1] <= srcBctrl[0];

	regDst  [1] <= regDst[0];

	zeroFlag [1]<= zeroFlag [0];

	writeData[1]<= writeData[0];

	readData[1] <= readData[0];

end

endmodule