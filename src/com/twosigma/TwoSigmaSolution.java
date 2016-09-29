package com.twosigma;

public class TwoSigmaSolution {

	public boolean isPowerOfFour(int num) {
		while(num>1){
			num/=4;
		}
		return num==1;
	}
	
	public boolean isPowerOfFour2(int num) {
		return num>0&&(num&(num-1))==0&&((num&0x55555555L)==num);
	}
	
	public static void main(String[] args){
		TwoSigmaSolution sol=new TwoSigmaSolution();
		System.out.println(sol.isPowerOfFour2(16));
	}
}
