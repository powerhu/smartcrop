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
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"os"
	"sync"

	pigo "github.com/esimov/pigo/core"
	"github.com/muesli/smartcrop"
	"github.com/muesli/smartcrop/nfnt"
)

var (
	qThresh float32 = 10.0
	cascadeFile = "./cascade/facefinder"
)

func main() {
	input := flag.String("input", "", "input filename")
	output := flag.String("output", "", "output filename")
	w := flag.Int("width", 0, "crop width")
	h := flag.Int("height", 0, "crop height")
	resize := flag.Bool("resize", true, "resize after cropping")
	quality := flag.Int("quality", 85, "jpeg quality")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "No input file given")
		os.Exit(1)
	}

	classifier, err := initClassifier(cascadeFile)

	if err != nil {
		fmt.Fprintf(os.Stderr, "fail to load cascade file:%s", cascadeFile)
		os.Exit(1)
	}

	// enumerateFolder(*input, *output, *w, *h, *resize, *quality)
	cropImage(*input, *output, *w, *h, *resize, *quality, classifier)
}

func enumerateFolder(inputDir string, outputDir string, w, h int, resize bool, quality int) {
	files, err := ioutil.ReadDir(inputDir)
	if err != nil {
		log.Fatal(err)
	}

	classifier, err := initClassifier(cascadeFile)

	if err != nil {
		fmt.Fprintf(os.Stderr, "fail to load cascade file:%s", cascadeFile)
		os.Exit(1)
	}

	var wg sync.WaitGroup
	for _, file := range files {
		if !file.IsDir() {
			var filename = file.Name()
			wg.Add(1)
			//go func() {
				fmt.Println("process:", filename)
				cropImage(inputDir + "/" + filename, outputDir +"/"+ filename, w, h, resize, quality, classifier)
				wg.Done()
			//}()
		}
	}

	wg.Wait()
}

func initClassifier(cascadeFile string) (*pigo.Pigo, error) {
	cascadeBytes, err := ioutil.ReadFile(cascadeFile)
	if err != nil {
		return nil, err
	}

	p := pigo.NewPigo()
	// Unpack the binary file. This will return the number of cascade trees,
	// the tree depth, the threshold and the prediction from tree's leaf nodes.
	classifier, err := p.Unpack(cascadeBytes)
	if err != nil {
		return nil, err
	}

	return classifier, nil
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

func cropImage(input string, output string, w, h int, resize bool, quality int, classifier *pigo.Pigo) {
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

	img = crop(img, w, h, resize, boosts)

	switch format {
	case "png":
		png.Encode(fOut, img)
	case "jpeg":
		jpeg.Encode(fOut, img, &jpeg.Options{Quality: quality})
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
