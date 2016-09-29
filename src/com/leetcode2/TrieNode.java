package com.leetcode2;

public class TrieNode {
	char c;
    TrieNode[] children = new TrieNode[26];
    boolean isLeaf;
    
    public TrieNode() {}
    public TrieNode(char c) {
        this.c = c;
    }
}
