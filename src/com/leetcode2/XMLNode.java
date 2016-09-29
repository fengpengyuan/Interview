package com.leetcode2;

import java.util.ArrayList;
import java.util.List;

public class XMLNode {
	String type;
	String val;
	List<XMLNode> children;
	
	public XMLNode(String type, String val){
		this.type=type;
		this.val=val;
		children=new ArrayList<XMLNode>();
	}
}
