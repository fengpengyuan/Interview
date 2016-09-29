package com.leetcode2;

public class FDGoogle {
	int numOfSpaces;
	String fileName;
	boolean isImage;
	String[] formats={"jpeg", "png", "gif"};
	
	public FDGoogle(String file){
		this.fileName=file;
		this.numOfSpaces=file.lastIndexOf(' ')+1;
		for(String s: formats){
			if(file.indexOf(s)!=-1)
				isImage=true;
		}
	}
}
