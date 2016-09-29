package com.leetcode2;

import java.util.Arrays;

public class NumMatrix {
//	int[][] sum;
//    public NumMatrix(int[][] matrix) {
//        int m=matrix.length;
//        if(m==0)
//            return;
//        int n=matrix[0].length;
//        sum=new int[m][n];
//        sum[0][0]=matrix[0][0];
//        for(int i=1;i<m;i++){
//            sum[i][0]=sum[i-1][0]+matrix[i][0];
//        }
//        
//        for(int i=1;i<n;i++){
//            sum[0][i]=sum[0][i-1]+matrix[0][i];
//        }
//        
//        for(int i=1;i<m;i++){
//            for(int j=1;j<n;j++){
//                sum[i][j]=sum[i-1][j]+sum[i][j-1]+matrix[i][j]-sum[i-1][j-1];
//            }
//        }
//        
//        for(int i=0;i<m;i++){
//        	System.out.println(Arrays.toString(sum[i]));
//        }
//    }
//
//    public int sumRegion(int row1, int col1, int row2, int col2) {
//    	if(row1==0&&col1==0)
//    		return sum[row2][col2];
//    	if(row1==0)
//    		return sum[row2][col2]-sum[row2][col1-1];
//    	if(col1==0)
//    		return sum[row2][col2]-sum[row1-1][col2];
//        return sum[row2][col2]-sum[row2][col1-1]-sum[row1-1][col2]+sum[row1-1][col1-1];
//    }
	
	int[][] sum;
    public NumMatrix(int[][] matrix) {
        int m=matrix.length;
        if(m==0)
            return;
        int n=matrix[0].length;
        sum=new int[m+1][n+1];
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                sum[i+1][j+1]=sum[i][j+1]+sum[i+1][j]+matrix[i][j]-sum[i][j];
            }
        }
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        
        return sum[row2+1][col2+1]-sum[row2+1][col1]-sum[row1][col2+1]+sum[row1][col1];
    }
    
    public static void main(String[] strs){
    	int[][] matrix={{3,0,1,4,2},
    			{5,6,3,2,1},
    			{1,2,0,1,5},
    			{4,1,0,1,7},
    			{1,0,3,0,5}};
    	
    	NumMatrix numMatrix = new NumMatrix(matrix);
    	numMatrix.sumRegion(2,1,4,3);
    	numMatrix.sumRegion(1,1,2,2);
    	numMatrix.sumRegion(1,2,2,4);
    }
}
