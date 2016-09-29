package com.leetcode2;

public class Trie {
//	private TrieNode root;
//
//    public Trie() {
//        root = new TrieNode();
//    }
//
//    // Inserts a word into the trie.
//    public void insert(String word) {
//        TrieNode node=root;
//        for(int i=0;i<word.length();i++){
//            TrieNode[] children=node.children;
//            char c=word.charAt(i);
//            if(children[c-'a']==null){
//                node=new TrieNode(c);
//                children[c-'a']=node;
//            } else{
//                node=children[c-'a'];
//            }
//        }
//        node.isLeaf=true;
//    }
//
//    // Returns if the word is in the trie.
//    public boolean search(String word) {
//        TrieNode node=root;
//        for(int i=0;i<word.length();i++){
//            TrieNode[] children=node.children;
//            char c=word.charAt(i);
//            if(children[c-'a']==null)
//                return false;
//            else
//                node=children[c-'a'];
//        }
//        return node.isLeaf;
//    }
//
//    // Returns if there is any word in the trie
//    // that starts with the given prefix.
//    public boolean startsWith(String prefix) {
//        TrieNode node=root;
//        for(int i=0;i<prefix.length();i++){
//            TrieNode[] children=node.children;
//            char c=prefix.charAt(i);
//            if(children[c-'a']==null)
//                return false;
//            else
//                node=children[c-'a'];
//        }
//        return true;
//    }
	
	public TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    // Inserts a word into the trie.
    public void insert(String word) {
        TrieNode node=root;
        for(int i=0;i<word.length();i++){
            TrieNode[] children=node.children;
            char c=word.charAt(i);
            if(children[c-'a']==null){
                node=new TrieNode(c);
                children[c-'a']=node;
            } else{
                node=children[c-'a'];
            }
        }
        node.isLeaf=true;
    }

    // Returns if the word is in the trie.
    public boolean search(String word) {
        TrieNode node=searchWord(word);
        return node==null?false:node.isLeaf;
    }

    // Returns if there is any word in the trie
    // that starts with the given prefix.
    public boolean startsWith(String prefix) {
        TrieNode node=searchWord(prefix);
        return node==null?false:true;
    }
    
    public TrieNode searchWord(String word){
        TrieNode node=root;
        for(int i=0;i<word.length();i++){
            TrieNode[] children=node.children;
            char c=word.charAt(i);
            if(children[c-'a']==null)
                return null;
            else
                node=children[c-'a'];
        }
        return node;
    }
}
