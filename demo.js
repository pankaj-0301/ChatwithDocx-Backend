const express = require("express");
const multer = require("multer");
const { DocxLoader } = require("@langchain/community/document_loaders/fs/docx");
const { PDFLoader } =require( "@langchain/community/document_loaders/fs/pdf");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");

const fs = require("fs");
const path = require("path");
const mongoose = require("mongoose");

const app = express();
const PORT = 5000;

const mongoURI = "mongodb+srv://pankaj:o9phHPzQ0p5D5Rq1@cluster0.yvs1pu5.mongodb.net/VECTORDB"; // Update with your database name
mongoose.connect(mongoURI).then(() => console.log("MongoDB connected"))
  .catch(err => console.error("MongoDB connection error:", err));

// Define a Mongoose schema
const chunkSchema = new mongoose.Schema({
  fileName: { type: String, required: true },
  chunk: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

const Chunk = mongoose.model("Chunk", chunkSchema);

// Configure multer for file uploads
const upload = multer({ dest: "uploads/" });

// Endpoint to handle file uploads, parsing, and splitting
app.post("/upload", upload.array("files", 20), async (req, res) => {
    try {
      const chunksArray = [];

      // Process each uploaded file
      for (const file of req.files) {
        const filePath = path.join(__dirname, file.path);
        console.log("File path:", filePath);

        let loader;

        // Step 1: Determine the file type (DOCX or PDF) and use the appropriate loader
        if (path.extname(file.originalname).toLowerCase() === ".docx") {
          loader = new DocxLoader(filePath);
          console.log("Loading DOCX document...");
        } else if (path.extname(file.originalname).toLowerCase() === ".pdf") {
          loader = new PDFLoader(filePath);
          console.log("Loading PDF document...");
        } else {
          // If the file format is not supported, skip it
          console.log("Unsupported file format:", file.originalname);
          continue;
        }

        // Load the document
        const docs = await loader.load();
        console.log("Document loaded:", docs);

        let content = "";
        if (docs.length > 0) {
          content = docs[0].pageContent; // Extract text content from the document
          console.log("Document content:", content);
        }

        // Cleanup: delete the uploaded file after processing
        fs.unlinkSync(filePath);
        console.log("File deleted.");

        // Step 2: Use a splitter to break the content into chunks
        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: 1000, // Define your chunk size here
          chunkOverlap: 200, // Optional overlap between chunks
        });

        console.log("Splitting document...");
        const chunks = await splitter.splitText(content);
        console.log("Chunks:", chunks);

        // Store each chunk in the MongoDB
        for (const chunk of chunks) {
          const newChunk = new Chunk({
            fileName: file.originalname,
            chunk: chunk
          });
          await newChunk.save();
          console.log(`Chunk from ${file.originalname} saved.`);
        }

        // Push the chunks for this file into the result array
        chunksArray.push({
          fileName: file.originalname,
          chunks,
        });
      }

      // Return all chunks from all files as a JSON response
      res.json({ files: chunksArray });
    } catch (error) {
      console.error("Error processing the documents:", error);
      res.status(500).send("Failed to process the files");
    }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
