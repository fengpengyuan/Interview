package com.lintcode;

import java.util.HashMap;

public class TrieNode {
	char c;
    boolean isLeaf;
    HashMap<Character,TrieNode> children;
    
    public TrieNode(){
        isLeaf=false;
        children=new HashMap<Character,TrieNode>();
    }
    
    public TrieNode(char c){
        this.c=c;
        isLeaf=false;
        children=new HashMap<Character,TrieNode>();
    }
}
