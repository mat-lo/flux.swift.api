// main.swift
import Vapor

// Configure the application
func configure(_ app: Application) throws {
    // Configure server settings
    app.http.server.configuration.hostname = "0.0.0.0"
    app.http.server.configuration.port = 8080
    
    // Configure maximum body size (10MB)
    app.routes.defaultMaxBodySize = "10mb"
    
    print("🚀 Starting FLUX API server...")
    print("📍 Server will be available at http://localhost:8080")
    print("💡 Try sending a POST request to /generate")
    
    // Register routes
    try routes(app)
    
    print("✅ Routes configured successfully")
}

// Create and run the application
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
