/*
 * Copyright (c) 2014-2019 Christian Muehlhaeuser
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *	Authors:
 *		Christian Muehlhaeuser <muesli@gmail.com>
 *		Michael Wendland <michael@michiwend.com>
 *		Bjørn Erik Pedersen <bjorn.erik.pedersen@gmail.com>
 *		Patryk Pomykalski <pomyks@gmail.com>
 */

package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/disintegration/imaging"
	"google.golang.org/grpc"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"os"
	fp "path/filepath"
	"sync"
	"time"

	pigo "github.com/esimov/pigo/core"
	"github.com/muesli/smartcrop"
	fd "github.com/muesli/smartcrop/facedetection"
	"github.com/muesli/smartcrop/nfnt"
)

var (
	qThresh float32 = 10.0
	cascadeFile = "./cascade/facefinder"
	address     = "localhost:50051"
	defaultName = "face"
)

type faceDetFunc func(string) ([]smartcrop.BoostRegion, error)

func loadData(file string) ([]byte, error) {
	return ioutil.ReadFile(file) // just pass the file name
}

func faceDetection(file string) ([]smartcrop.BoostRegion, error) {
	rawImage, err := loadData(file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fail to load raw image file:%s", file)
		return nil, err
	}

	// Set up a connection to the server.
	conn, err := grpc.Dial(address, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
		return nil, err
	}
	defer conn.Close()

	c := fd.NewFaceDetServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	request := &fd.FaceDetRequest{
		ImageData: rawImage,
		Type: "jpeg",
		ConfThresh: 0.4,
	}

	resp, err := c.Predict(ctx, request)

	if err != nil {
		log.Fatalf("error when predict: %v", err)
		return nil, err
	}

	var boosts []smartcrop.BoostRegion
	for _, det := range resp.DetObjs {
		boosts = append(boosts, smartcrop.BoostRegion {
			X: int(det.Lx),
			Y: int(det.Ly),
			Width: int(det.Rx - det.Lx),
			Height: int(det.Ry - det.Ly),
			Weight: 1.0,
		})
	}

	return boosts, err
}

func main() {
	input := flag.String("input", "", "input filename")
	output := flag.String("output", "", "output filename")
	w := flag.Int("width", 0, "crop width")
	h := flag.Int("height", 0, "crop height")
	resize := flag.Bool("resize", true, "resize after cropping")
	enableCenter := flag.Bool("center", true, "enable auto center crop")
	faceDetApi := flag.Bool("api", true, "use third-party api to do face detection")
	quality := flag.Int("quality", 85, "jpeg quality")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "No input file given")
		os.Exit(1)
	}


	//enumerateFolder(*input, *output, *w, *h, *resize, *quality)

	if *faceDetApi {
		cropImage(*input, *output, *w, *h, *resize, *quality, *enableCenter, faceDetection)
	} else {
		openCVFaceCall := initOpenCvFaceClassifier(cascadeFile)

		cropImage(*input, *output, *w, *h, *resize, *quality, *enableCenter, openCVFaceCall)
	}
}

func enumerateFolder(inputDir string, outputDir string, w, h int, resize bool, quality int) {
	files, err := ioutil.ReadDir(inputDir)
	if err != nil {
		log.Fatal(err)
	}

	// Set up a connection to the server.
	conn, err := grpc.Dial(address, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("did not connect: %v", err)
		return
	}
	defer conn.Close()

	c := fd.NewFaceDetServiceClient(conn)

	var wg sync.WaitGroup
	for _, file := range files {
		ext := fp.Ext(file.Name())
		if !file.IsDir() && (ext ==".jpg" || ext == ".png" || ext == ".jpeg") {
			var filename = file.Name()
			wg.Add(1)
			//go func() {
				fmt.Println("process:", filename)
				cropImage(inputDir + "/" + filename, outputDir +"/"+ filename, w, h, resize, quality, true,
					func(file string) ([]smartcrop.BoostRegion, error) {
						rawImage, err := loadData(file)
						if err != nil {
							fmt.Fprintf(os.Stderr, "fail to load raw image file:%s", file)
							return nil, err
						}
						request := &fd.FaceDetRequest{
							ImageData: rawImage,
							Type: "jpeg",
							ConfThresh: 0.4,
						}
						ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
						defer cancel()

						resp, err := c.Predict(ctx, request)

						if err != nil {
							log.Fatalf("error when predict: %v", err)
							return nil, err
						}

						var boosts []smartcrop.BoostRegion
						for _, det := range resp.DetObjs {
							boosts = append(boosts, smartcrop.BoostRegion {
								X: int(det.Lx),
								Y: int(det.Ly),
								Width: int(det.Rx - det.Lx),
								Height: int(det.Ry - det.Ly),
								Weight: 1.0,
							})
						}

						return boosts, err
				})
				wg.Done()
			//}()
		}
	}

	wg.Wait()
}

func initOpenCvFaceClassifier(cascadeFile string) func(file string) ([]smartcrop.BoostRegion, error) {
	cascadeBytes, err := ioutil.ReadFile(cascadeFile)
	if err != nil {
		return nil
	}

	p := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := p.Unpack(cascadeBytes)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fail to load cascade file:%s", cascadeFile)
		return nil
	}

	openCvFace := func(file string) ([]smartcrop.BoostRegion, error) {
		f, err := os.Open(file)
		if err != nil {
			fmt.Fprintf(os.Stderr, "can't open input file: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()

		img, _, err := image.Decode(f)
		if err != nil {
			fmt.Fprintf(os.Stderr, "can't decode input file: %v\n", err)
			return nil, err
		}

		faces := faceDet(img, classifier)

		// convert dets into boost region
		var boosts []smartcrop.BoostRegion
		for _, face := range faces {
			if face.Q > qThresh {
				boosts = append(boosts, smartcrop.BoostRegion {
					X: face.Col - face.Scale/2,
					Y: face.Row -face.Scale/2,
					Width: face.Scale,
					Height: face.Scale,
					Weight: 1.0,
				})
			}
		}

		return boosts, nil
	}

	return openCvFace
}

func faceDet(src image.Image, classifier *pigo.Pigo) []pigo.Detection  {
	pixels := pigo.RgbToGrayscale(src)
	cols, rows := src.Bounds().Max.X, src.Bounds().Max.Y

	cParams := pigo.CascadeParams{
		MinSize:     20,
		MaxSize:     1000,
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,

		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	angle := 0.0 // cascade rotation angle. 0.0 is 0 radians and 1.0 is 2*pi radians

	// Run the classifier over the obtained leaf nodes and return the detection results.
	// The result contains quadruplets representing the row, column, scale and detection score.
	dets := classifier.RunCascade(cParams, angle)

	// Calculate the intersection over union (IoU) of two clusters.
	dets = classifier.ClusterDetections(dets, 0.2)

	return dets
}

func cropImage(input string, output string, w, h int, resize bool, quality int, enableCenter bool, faceCall faceDetFunc) {
	f, err := os.Open(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "can't open input file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	img, format, err := image.Decode(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "can't decode input file: %v\n", err)
		os.Exit(1)
	}

	boosts, err := faceCall(input)

	out := output
	var fOut io.WriteCloser
	if out == "-" {
		fOut = os.Stdout
	} else {
		fOut, err = os.Create(out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "can't create output file: %v\n", err)
			os.Exit(1)
		}
		defer fOut.Close()
	}

	oriRatio := float64(img.Bounds().Dx()) / float64(img.Bounds().Dy())
	wantRatio := float64(w) / float64(h)

	var cbImg image.Image
	if enableCenter && oriRatio >= wantRatio && oriRatio <= wantRatio * 1.4 {
		cbImg = centerCrop(img, w, h, 100.0, resize)
	} else {
		cbImg = crop(img, w, h, resize, boosts)
	}

	// var imageList []*image.RGBA
	// imageList = append(imageList, smartcrop.ToRGBA(img))
	// newImg := crop(img, w, h, resize, boosts)
	// imageList = append(imageList, smartcrop.ToRGBA(newImg))
	// cbImg := smartcrop.CombineImage(imageList)

	switch format {
	case "png":
		png.Encode(fOut, cbImg)
	case "jpeg":
		jpeg.Encode(fOut, cbImg, &jpeg.Options{Quality: quality})
	}
}

func crop(img image.Image, w, h int, resize bool, boosts []smartcrop.BoostRegion) image.Image {
	width, height := getCropDimensions(img, w, h)
	resizer := nfnt.NewDefaultResizer()
	analyzer := smartcrop.NewAnalyzer(resizer)
	topCrop, _ := analyzer.FindBestCrop(img, width, height, boosts)

	type SubImager interface {
		SubImage(r image.Rectangle) image.Image
	}

	img = img.(SubImager).SubImage(topCrop)
	if resize && (img.Bounds().Dx() != width || img.Bounds().Dy() != height) {
		img = resizer.Resize(img, uint(width), uint(height))
	}
	return img
}

func getCropDimensions(img image.Image, width, height int) (int, int) {
	// if we don't have width or height set use the smaller image dimension as both width and height
	if width == 0 && height == 0 {
		bounds := img.Bounds()
		x := bounds.Dx()
		y := bounds.Dy()
		if x < y {
			width = x
			height = x
		} else {
			width = y
			height = y
		}
	}
	return width, height
}

func centerCropImage(input string, output string, w, h int, resize bool, quality int) {
	f, err := os.Open(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "can't open input file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	img, format, err := image.Decode(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "can't decode input file: %v\n", err)
		os.Exit(1)
	}

	out := output
	var fOut io.WriteCloser
	if out == "-" {
		fOut = os.Stdout
	} else {
		fOut, err = os.Create(out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "can't create output file: %v\n", err)
			os.Exit(1)
		}
		defer fOut.Close()
	}

	newImg := centerCrop(img, w, h, 100.0, resize)

	switch format {
	case "png":
		png.Encode(fOut, newImg)
	case "jpeg":
		jpeg.Encode(fOut, newImg, &jpeg.Options{Quality: quality})
	}
}

func Round(a float64) int {
	b := int(a)
	d := a - float64(b)
	if d >= 0.5 || d <= -0.5 {
		return b + 1
	} else {
		return b
	}
}

func centerCrop(im image.Image, width, height int, sigma float64, resize bool) image.Image {
	resizer := nfnt.NewDefaultResizer()
	var startPoint image.Point
	w, h := im.Bounds().Dx(), im.Bounds().Dy()
	oriRatio := float64(w) / float64(h)
	wantRatio := float64(width) / float64(height)

	ratio := wantRatio / oriRatio
	var x, y int
	var ww, wh int
	if oriRatio > wantRatio {
		//mw := float64(w) * radio
		x = Round(float64(w) * ratio)
		y = h
		ww = w
		wh = Round(float64(ww) / wantRatio)
		startPoint.X = 0
		startPoint.Y = (wh - h) / 2
	} else {
		//mw := float64(w) * radio
		mh := float64(h) * ratio
		x = w
		y = Round(float64(h*h) / mh)
		wh = h
		ww = Round(float64(h) * wantRatio)
		startPoint.Y = 0
		startPoint.X = (ww - w) / 2
	}
	bg := imaging.CropCenter(im, x, y)
	bg = imaging.Resize(bg, ww, wh, imaging.Lanczos)
	bg = imaging.Blur(bg, sigma)

	bg = imaging.Paste(bg, im, startPoint)

	if resize && (bg.Bounds().Dx() != width || bg.Bounds().Dy() != height) {
		return resizer.Resize(image.Image(bg), uint(width), uint(height))
	}

	return bg
}
