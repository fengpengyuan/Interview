package com.leetcode;
import java.util.HashMap;

class TrieNode1 {
    // Initialize your data structure here.
    char c;
    HashMap<Character, TrieNode1> children;
    boolean isEnd;
    public TrieNode1() {
        children=new HashMap<Character, TrieNode1>();
        isEnd=false;
    }
}

public class Trie {
	private TrieNode1 root;

    public Trie() {
        root = new TrieNode1();
    }

    // Inserts a word into the trie.
    public void insert(String word) {
        if(word.length()==0)
            return;
        TrieNode1 node=root;
        for(int i=0;i<word.length();i++){
            char c=word.charAt(i);
            HashMap<Character, TrieNode1> children=node.children;
            if(children.containsKey(c))
                node=children.get(c);
            else{
                TrieNode1 trieNode=new TrieNode1();
                children.put(c, trieNode);
                node=trieNode;
            }
        }
        node.isEnd=true;
    }

    // Returns if the word is in the trie.
    public boolean search(String word) {
        TrieNode1 cur=root;
        for(int i=0;i<word.length();i++){
            char c=word.charAt(i);
            HashMap<Character, TrieNode1> children=cur.children;
            if(!children.containsKey(c))
            	return false;
            cur=children.get(c);
        }
        if(cur.isEnd)
        	return true;
        return false;
    }

    // Returns if there is any word in the trie
    // that starts with the given prefix.
    public boolean startsWith(String prefix) {
    	TrieNode1 cur=root;
        for(int i=0;i<prefix.length();i++){
            char c=prefix.charAt(i);
            HashMap<Character, TrieNode1> children=cur.children;
            if(!children.containsKey(c))
            	return false;
            cur=children.get(c);
        }
        return true;
    }
    
    
    public static void main(String[] args){
    	
    }
}

