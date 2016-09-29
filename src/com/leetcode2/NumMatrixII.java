package com.leetcode2;

public class NumMatrixII {
	int[][] matrix;
	int[][] colSums;
	
	public NumMatrixII(int[][] matrix){
		int m=matrix.length;
		if(m==0)
			return;
		int n=matrix[0].length;
		this.matrix=matrix;
		colSums=new int[m+1][n];
		
		for(int i=1;i<=m;i++){
			for(int j=0;j<n;j++){
				colSums[i][j]=colSums[i-1][j]+matrix[i-1][j];
			}
		}
	}
	
	public void update(int row, int col, int val) {
		for(int i=row+1;i<colSums.length;i++){
			colSums[i][col]=colSums[i][col]+val-matrix[i][col];
		}
		matrix[row][col]=val;
	}
	
	public int sumRegion(int row1, int col1, int row2, int col2) {
		int res=0;
		for(int i=col1; i<=col2;i++){
			res+=colSums[row2+1][i]-colSums[row1][i];
		}
		return res;
	}
}
