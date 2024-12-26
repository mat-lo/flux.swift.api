// main.swift
import Vapor

// Configure the application
func configure(_ app: Application) throws {
    print("🚀 Starting FLUX API server...")
    print("📍 Server will be available at http://localhost:8080")
    
    let workingDir = DirectoryConfiguration.detect().workingDirectory
    print("💡 Working directory: \(workingDir)")
    print("💾 Images directory: \(workingDir)Public/images/")
    
    // Configure server settings
    app.http.server.configuration.hostname = "0.0.0.0"
    app.http.server.configuration.port = 8080
    
    // Configure file middleware
    app.middleware.use(FileMiddleware(publicDirectory: workingDir + "Public/"))
    
    // Register routes
    try routes(app)
}

let app = Application(.development)
defer { app.shutdown() }

do {
    try configure(app)
    print("🎯 Server is ready to accept requests")
    try app.run()
} catch {
    print("❌ Error starting server: \(error)")
    exit(1)
}
