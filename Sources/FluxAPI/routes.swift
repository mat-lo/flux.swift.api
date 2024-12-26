// routes.swift
import Vapor
import FluxSwift
import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Progress
import Tokenizers
import CoreGraphics
import UniformTypeIdentifiers
import ImageIO

// Copy the Image struct from the original code
enum ImageError: Error {
    case failedToSave
    case unableToOpen
}

public struct Image {
    public let data: MLXArray

    public init(_ data: MLXArray) {
        precondition(data.ndim == 3)
        self.data = data
    }

    public init(url: URL, maximumEdge: Int? = nil) throws {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw ImageError.unableToOpen
        }

        self.init(image: image)
    }

    public init(image: CGImage, maximumEdge: Int? = nil) {
        var width = image.width
        var height = image.height

        if let maximumEdge {
            func scale(_ edge: Int, _ maxEdge: Int) -> Int {
                Int(round(Float(maximumEdge) / Float(maxEdge) * Float(edge)))
            }

            if width >= height {
                width = scale(width, image.width)
                height = scale(height, image.width)
            } else {
                width = scale(width, image.height)
                height = scale(height, image.height)
            }
        }

        width = width - width % 64
        height = height - height % 64

        var raster = Data(count: width * 4 * height)
        raster.withUnsafeMutableBytes { ptr in
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!
            let context = CGContext(
                data: ptr.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue)!

            context.draw(
                image, in: CGRect(origin: .zero, size: .init(width: width, height: height)))
        }

        self.data = MLXArray(raster, [height, width, 4], type: UInt8.self)[0..., 0..., ..<3]
    }

    public func asCGImage() -> CGImage {
        var raster = data

        if data.dim(-1) == 3 {
            raster = padded(raster, widths: [0, 0, [0, 1]])
        }

        class DataHolder {
            var data: Data
            init(_ data: Data) {
                self.data = data
            }
        }

        let holder = DataHolder(raster.asData())
        let payload = Unmanaged.passRetained(holder).toOpaque()
        
        func release(payload: UnsafeMutableRawPointer?, data: UnsafeMutableRawPointer?) {
            Unmanaged<DataHolder>.fromOpaque(payload!).release()
        }

        return holder.data.withUnsafeMutableBytes { ptr in
            let (H, W, C) = raster.shape3
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!

            let context = CGContext(
                data: ptr.baseAddress, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * C,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue, releaseCallback: release,
                releaseInfo: payload)!
            return context.makeImage()!
        }
    }

    public func save(url: URL) throws {
        let uti = UTType(filenameExtension: url.pathExtension) ?? UTType.png

        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, uti.identifier as CFString, 1, nil)
        else {
            throw ImageError.failedToSave
        }
        
        CGImageDestinationAddImage(destination, asCGImage(), nil)
        if !CGImageDestinationFinalize(destination) {
            throw ImageError.failedToSave
        }
    }
}

struct GenerateImageRequest: Content {
    var prompt: String
    
    var width: Int?
    var height: Int?
    var steps: Int?
    var guidance: Float?
    var model: String?
    var hfToken: String?
    var seed: UInt64?
    var quantize: Bool?
    var float16: Bool?
    var loraPath: String?
    var initImagePath: String?
    var initImageStrength: Float?
    
    var finalWidth: Int { width ?? 512 }
    var finalHeight: Int { height ?? 512 }
    var finalSteps: Int { steps ?? 4 }
    var finalGuidance: Float { guidance ?? 3.5 }
    var finalModel: String { model?.lowercased() ?? "schnell" }
    var finalQuantize: Bool { quantize ?? false }
    var finalFloat16: Bool { float16 ?? true }
}

func routes(_ app: Application) throws {
    app.post("generate") { req async throws -> Response in
        let request = try req.content.decode(GenerateImageRequest.self)
        
        // Set up model configuration
        let selectedModel: FluxConfiguration
        var token: String?
        let defaultTokenLocation = NSString("~/.cache/huggingface/token").expandingTildeInPath
        
        switch request.finalModel {
        case "schnell":
            selectedModel = FluxConfiguration.flux1Schnell
        case "dev":
            selectedModel = FluxConfiguration.flux1Dev
            token = request.hfToken
            
            if token == nil {
                token = try? String(contentsOfFile: defaultTokenLocation, encoding: .utf8)
                req.logger.info("Using default Hugging Face token from \(defaultTokenLocation)")
            }
        default:
            throw Abort(.badRequest, reason: "Invalid model type. Please choose 'schnell' or 'dev'.")
        }
        
        // Download model
        req.logger.info("Downloading or loading model...")
        try await selectedModel.download(hub: request.finalModel == "dev" ? HubApi(hfToken: token) : HubApi())
        
        let loadConfiguration = LoadConfiguration(
            float16: request.finalFloat16,
            quantize: request.finalQuantize,
            loraPath: request.loraPath
        )
        
        // Handle LoRA weights if specified
        if let loraPath = loadConfiguration.loraPath {
            if !FileManager.default.fileExists(atPath: loraPath) {
                req.logger.info("Downloading LoRA weights...")
                try await selectedModel.downloadLoraWeights(loadConfiguration: loadConfiguration)
            }
        }
        
        // Set up generator and parameters
        let generator: ImageGenerator?
        var denoiser: DenoiseIterator?
        
        let parameters = EvaluateParameters(
            width: request.finalWidth,
            height: request.finalHeight,
            numInferenceSteps: request.finalSteps,
            guidance: request.finalGuidance,
            seed: request.seed,
            prompt: request.prompt,
            shiftSigmas: request.finalModel == "dev"
        )
        
        req.logger.info("Initializing generator...")
        generator = try selectedModel.textToImageGenerator(configuration: loadConfiguration)
        generator?.ensureLoaded()
        denoiser = (generator as? TextToImageGenerator)?.generateLatents(parameters: parameters)
        
        // Generate image
        req.logger.info("Starting image generation...")
        var lastXt: MLXArray!
        while let xt = denoiser!.next() {
            req.logger.info("Step \(denoiser!.i)/\(parameters.numInferenceSteps)")
            eval(xt)
            lastXt = xt
        }
        
        // Process generated image
        req.logger.info("Processing generated image...")
        let unpackedLatents = unpackLatents(lastXt, height: parameters.height, width: parameters.width)
        let decoded = generator?.decode(xt: unpackedLatents)
        let imageData = decoded?.squeezed()
        let raster = (imageData! * 255).asType(.uint8)
        
        // Create Image instance
        let image = Image(raster)
        
        // Create temporary file and save the image
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent(UUID().uuidString + ".png")
        
        // Save image to temporary file
        try image.save(url: tempFile)
        
        // Read the PNG file
        let pngData = try Data(contentsOf: tempFile)
        
        // Clean up
        try? FileManager.default.removeItem(at: tempFile)
        
        // Create response
        req.logger.info("Sending response...")
        let response = Response(status: .ok)
        response.headers.contentType = .png
        response.body = .init(data: pngData)
        return response
    }
}

// Helper function for unpacking latents
private func unpackLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
    let reshaped = latents.reshaped(1, height / 16, width / 16, 16, 2, 2)
    let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
    return transposed.reshaped(1, height / 16 * 2, width / 16 * 2, 16)
}
