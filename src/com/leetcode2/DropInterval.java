package com.leetcode2;

public class DropInterval {
	double left, right;
	public DropInterval(double left, double right){
		this.left=left;
		this.right=right;
	}
	
	public boolean isWet(){
		return left>=right;
	}
}
